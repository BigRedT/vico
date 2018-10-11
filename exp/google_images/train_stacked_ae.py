import os
import h5py
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from tensorboard_logger import configure, log_value
import numpy as np
from PIL import Image

import utils.io as io
from utils.html_writer import HtmlWriter
from utils.model import Model
from utils.constants import save_constants
import utils.pytorch_layers as pytorch_layers
from exp.google_images.dataset_image_level import GoogleImagesImageLevelDataset
from exp.google_images.models.resnet_encoder_outer import resnet34
from exp.google_images.models.resnet_encoder_inner import ResnetEncoderInner
from exp.google_images.models.resnet_decoder_outer import ResnetDecoderOuter
from exp.google_images.models.resnet_decoder_inner import ResnetDecoderInner
from exp.google_images.models.word_classification_mlp import WordClassifierLayer


def l2_norm_sq(x):
    return torch.mean(torch.sum(x*x,1))


def train_model(exp_const,dataloader,model):   
    outer_opt_params = itertools.chain(        
        model.encoder_outer.parameters(),
        model.decoder_outer.parameters())
    inner_opt_params = itertools.chain(        
        model.word_classifier_layers.parameters(),
        model.encoder_inner.parameters(),
        model.decoder_inner.parameters())    

    if exp_const.optimizer == 'SGD':
        outer_opt = optim.SGD(
            outer_opt_params,
            lr=exp_const.lr,
            momentum=exp_const.momentum)
        inner_opt = optim.SGD(
            inner_opt_params,
            lr=exp_const.lr,
            momentum=exp_const.momentum)
    else:
        assert(False), 'optimizer not implemented'
    
    word_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    if exp_const.load_finetuned is True:
        step = model.const.model_num
    else:
        step = -1

    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloader):
            step += 1
            
            regions_unnormalized = \
                Variable(torch.FloatTensor(data['img']).cuda())
            regions = dataloader.dataset.normalize(
                regions_unnormalized/255,
                model.img_mean,
                model.img_std)
            regions = regions.permute(0,3,1,2)
            regions_unnormalized = regions_unnormalized.permute(0,3,1,2)
            word_idx = Variable(torch.LongTensor(data['idx']).cuda())
            
            resnet_feats = model.encoder_outer(regions)
            
            # Outer update
            recon_outer = model.decoder_outer(resnet_feats)
            recon_outer_loss = model.decoder_outer.compute_loss(
                0.5*(recon_outer+1),
                regions_unnormalized/255)
            outer_loss = recon_outer_loss

            outer_opt.zero_grad()
            outer_loss.backward()
            for p in outer_opt_params:
                p.grad = torch.clamp(p.grad,-0.1,0.1)
            outer_opt.step()

            # Inner update
            x, x_norm = model.encoder_inner(Variable(resnet_feats.data))
            embedding = x_norm
            recon_inner = model.decoder_inner(embedding)
            recon_outer_from_recon_inner = model.decoder_outer(recon_inner)
            recon_inner_loss = \
                model.decoder_inner.compute_loss(
                    recon_inner,
                    resnet_feats)
            recon_outer_from_recon_inner_loss = \
                model.decoder_outer.compute_loss(
                    0.5*(recon_outer_from_recon_inner+1),
                    regions_unnormalized/255)
            word_scores = model.word_classifier_layers(embedding)
            word_loss = word_criterion(
                word_scores,
                word_idx)
            feat_l2_norm_sq = l2_norm_sq(x)
            inner_loss = \
                recon_inner_loss + \
                recon_outer_from_recon_inner_loss + \
                0*word_loss + \
                (1e-6)*feat_l2_norm_sq
            
            inner_opt.zero_grad()
            inner_loss.backward()
            for p in inner_opt_params:
                p.grad = torch.clamp(p.grad,-0.1,0.1)
            inner_opt.step()

            total_loss = outer_loss + inner_loss
            
            if step%10==0:
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    'Total Loss: {:.4f} | ' + \
                    'Word Loss: {:.4f} | ' + \
                    'Recon Inner Loss: {:.4f} | ' + \
                    'Recon Outer Loss: {:.4f} | ' + \
                    'Recon Outer From Recon Inner Loss: {:.4f} | ' + \
                    'Feat L2: {:.4f}'
                log_str = log_str.format(
                    epoch,
                    it,
                    step,
                    total_loss.data[0],
                    word_loss.data[0],
                    recon_inner_loss.data[0],
                    recon_outer_loss.data[0],
                    recon_outer_from_recon_inner_loss.data[0],
                    feat_l2_norm_sq.data[0])
                print(log_str)
                log_value('Total Loss',total_loss.data[0],step)
                log_value('Word Loss',word_loss.data[0],step)
                log_value('Recon Inner Loss',recon_inner_loss.data[0],step)
                log_value('Recon Outer Loss',recon_outer_loss.data[0],step)
                log_value(
                    'Recon Outer from Inner Loss',
                    recon_outer_from_recon_inner_loss.data[0],
                    step)
                log_value('Feat L2 Sq Loss',feat_l2_norm_sq.data[0],step)

            if step%100==0:
                print(exp_const.exp_name)
                
            if step%1000==0:
                encoder_outer_path = os.path.join(
                    exp_const.model_dir,
                    f'encoder_outer_{step}')
                encoder_inner_path = os.path.join(
                    exp_const.model_dir,
                    f'encoder_inner_{step}')
                decoder_outer_path = os.path.join(
                    exp_const.model_dir,
                    f'decoder_outer_{step}')
                decoder_inner_path = os.path.join(
                    exp_const.model_dir,
                    f'decoder_inner_{step}')
                word_classifier_layers_path = os.path.join(
                    exp_const.model_dir,
                    f'word_classifier_layers_{step}')
                torch.save(
                    model.encoder_outer.state_dict(),
                    encoder_outer_path)
                torch.save(
                    model.encoder_inner.state_dict(),
                    encoder_inner_path)
                torch.save(
                    model.decoder_outer.state_dict(),
                    decoder_outer_path)
                torch.save(
                    model.decoder_inner.state_dict(),
                    decoder_inner_path)
                torch.save(
                    model.word_classifier_layers.state_dict(),
                    word_classifier_layers_path)
                

            if step%200==0:
                visualize(
                    255*0.5*(1+recon_outer_from_recon_inner),
                    255*0.5*(1+recon_outer),
                    regions_unnormalized,
                    exp_const.vis_dir)                        


def visualize(recon,recon_gt,gt,outdir):
    recon = recon.permute(0,2,3,1).data.cpu().numpy()
    recon_gt = recon_gt.permute(0,2,3,1).data.cpu().numpy()
    gt = gt.permute(0,2,3,1).data.cpu().numpy()
    html_filename = os.path.join(outdir,'index.html')
    html_writer = HtmlWriter(html_filename)
    B = recon.shape[0]
    for i in range(B):
        try:
            recon_img = Image.fromarray(recon[i].astype(np.uint8))
            filename = os.path.join(outdir,f'recon_{i}.png')
            recon_img.save(filename)
        except:
            import pdb; pdb.set_trace()

        try:
            recon_gt_img = Image.fromarray(recon_gt[i].astype(np.uint8))
            filename = os.path.join(outdir,f'recon_gt_{i}.png')
            recon_gt_img.save(filename)
        except:
            import pdb; pdb.set_trace()

        try:
            gt_img = Image.fromarray(gt[i].astype(np.uint8))
            filename = os.path.join(outdir,f'gt_{i}.png')
            gt_img.save(filename)
        except:
            import pdb; pdb.set_trace()


        col_dict = {
            0: html_writer.image_tag(f'gt_{i}.png',height=224,width=224),
            1: html_writer.image_tag(f'recon_gt_{i}.png',height=224,width=224),
            2: html_writer.image_tag(f'recon_{i}.png',height=224,width=224),
        }

        html_writer.add_element(col_dict)

    html_writer.close()


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    io.mkdir_if_not_exists(exp_const.vis_dir)
    configure(exp_const.log_dir)
    save_constants({'exp': exp_const,'data': data_const},exp_const.exp_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.encoder_outer = resnet34(pretrained=True)
    model.encoder_inner = ResnetEncoderInner(model.const.encoder_inner)
    model.decoder_outer = ResnetDecoderOuter(model.const.decoder_outer)
    model.decoder_inner = ResnetDecoderInner(model.const.decoder_inner)
    model.word_classifier_layers = WordClassifierLayer(
        model_const.word_classifier_layers)

    if exp_const.load_finetuned == True:
        model.encoder_outer.load_state_dict(
            torch.load(model.const.encoder_outer_path))
        model.encoder_inner.load_state_dict(
            torch.load(model.const.encoder_inner_path))
        model.word_classifier_layers.load_state_dict(torch.load(
            model.const.word_classifier_layers_path))
        if exp_const.use_recon_loss == True:
            model.decoder_outer.load_state_dict(
                torch.load(model.const.decoder_outer_path))
            model.decoder_inner.load_state_dict(
                torch.load(model.const.decoder_inner_path))
    
    model.encoder_outer.cuda()
    model.encoder_inner.cuda()
    model.encoder_outer.train()
    model.encoder_inner.train()
    model.decoder_outer.cuda()
    model.decoder_inner.cuda()
    model.decoder_outer.train()
    model.decoder_inner.train()
    model.word_classifier_layers.cuda()
    model.word_classifier_layers.train()

    model.img_mean = Variable(
        torch.FloatTensor(np.array([0.485, 0.456, 0.406]))).cuda()
    model.img_std = Variable(
        torch.FloatTensor(np.array([0.229, 0.224, 0.225]))).cuda()
    model.to_file(os.path.join(exp_const.exp_dir,'model.txt'))

    print('Creating dataloader ...')
    dataset = GoogleImagesImageLevelDataset(data_const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    train_model(exp_const,dataloader,model)