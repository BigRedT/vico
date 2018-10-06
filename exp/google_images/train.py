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
from exp.google_images.models.resnet import resnet152
from exp.google_images.models.resnet_normalized import resnet152_normalized
from exp.google_images.models.word_classification_mlp import WordClassifierLayer
from exp.google_images.models.decoder import Decoder


def l2_norm_sq(x):
    return torch.mean(torch.sum(x*x,1))


def train_model(exp_const,dataloader,model):   
    if exp_const.train_net==True:
        if exp_const.use_recon_loss==True:
            params = itertools.chain(        
                model.word_classifier_layers.parameters(),
                model.net.parameters(),
                model.decoder.parameters())
        else:
            params = itertools.chain(        
                model.word_classifier_layers.parameters(),
                model.net.parameters())
    else:
        if exp_const.use_recon_loss==True:
            params = itertools.chain(        
                model.word_classifier_layers.parameters(),
                model.decoder.parameters())
        else:
            params = itertools.chain(        
                model.word_classifier_layers.parameters())

    if exp_const.optimizer == 'SGD':
        opt = optim.SGD(
            params,
            lr=exp_const.lr,
            momentum=exp_const.momentum)
    elif exp_const.optimizer == 'Adam':
        opt = optim.Adam(
            params,
            lr=exp_const.lr)
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
            
            if exp_const.use_resnet_normalized == True:
                out, last_layer_features, last_layer_features_ = \
                    model.net(regions)
            else:
                out, last_layer_features = model.net(regions)
                
            word_scores = model.word_classifier_layers(
                last_layer_features)
            word_loss = word_criterion(
                word_scores,
                word_idx)
            feat_l2_norm_sq = l2_norm_sq(last_layer_features_)
            
            loss = word_loss
            if exp_const.use_resnet_normalized == True:
                loss = loss + (1e-6)*feat_l2_norm_sq
            
            recon_loss = Variable(torch.FloatTensor([0])).cuda()
            if exp_const.use_recon_loss == True:
                recon_output = model.decoder(last_layer_features)
                recon_loss = model.decoder.recon_loss(
                        0.5*(recon_output+1),
                        regions_unnormalized/255.)
                loss = loss + recon_loss
            
            opt.zero_grad()
            loss.backward()
            for p in params:
                p.grad = torch.clamp(p.grad,-0.1,0.1)
            opt.step()
            
            if step%10==0:
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    'Total Loss: {:.4f} | ' + \
                    'Word Loss: {:.4f} | ' + \
                    'Recon Loss: {:.4f} | ' + \
                    'Feat L2: {:.4f}'
                log_str = log_str.format(
                    epoch,
                    it,
                    step,
                    loss.data[0],
                    word_loss.data[0],
                    recon_loss.data[0],
                    feat_l2_norm_sq.data[0])
                print(log_str)
                log_value('Total Loss',loss.data[0],step)
                log_value('Word Loss',word_loss.data[0],step)
                log_value('Recon Loss',recon_loss.data[0],step)
                log_value('Feat L2 Sq Loss',feat_l2_norm_sq.data[0],step)

            if step%100==0:
                print(exp_const.exp_name)
                
            if step%1000==0:
                net_path = os.path.join(
                    exp_const.model_dir,
                    f'net_{step}')
                word_classifier_layers_path = os.path.join(
                    exp_const.model_dir,
                    f'word_classifier_layers_{step}')
                decoder_path = os.path.join(
                    exp_const.model_dir,
                    f'decoder_{step}')
                torch.save(model.net.state_dict(),net_path)
                torch.save(
                    model.word_classifier_layers.state_dict(),
                    word_classifier_layers_path)
                torch.save(
                    model.decoder.state_dict(),
                    decoder_path)
                

            if exp_const.use_recon_loss == True:
                if step%200==0:
                    visualize(
                        255*0.5*(1+recon_output),
                        regions_unnormalized,
                        exp_const.vis_dir)                        


def visualize(recon,gt,outdir):
    recon = recon.permute(0,2,3,1).data.cpu().numpy()
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
            gt_img = Image.fromarray(gt[i].astype(np.uint8))
            filename = os.path.join(outdir,f'gt_{i}.png')
            gt_img.save(filename)
        except:
            import pdb; pdb.set_trace()


        col_dict = {
            0: html_writer.image_tag(f'gt_{i}.png',height=224,width=224),
            1: html_writer.image_tag(f'recon_{i}.png',height=224,width=224),
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
    if exp_const.use_resnet_normalized == True:
        model.net = resnet152_normalized(pretrained=True)
    else:
        model.net = resnet152(pretrained=True)
    model.word_classifier_layers = WordClassifierLayer(
        model_const.word_classifier_layers)
    if exp_const.use_recon_loss == True:
        model.decoder = Decoder(model.const.decoder).cuda()
        model.decoder.train()
    if exp_const.load_finetuned == True:
        model.net.load_state_dict(torch.load(model.const.net_path))
        model.word_classifier_layers.load_state_dict(torch.load(
            model.const.word_classifier_layers_path))
    model.net.cuda()
    model.word_classifier_layers.cuda()
    model.net.train()
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