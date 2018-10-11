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


def vis_recon(exp_const,dataloader,model):   
    for step,data in enumerate(dataloader):
        if step >= exp_const.max_iter:
            break

        regions_unnormalized = \
            Variable(torch.FloatTensor(data['img']).cuda(),volatile=True)
        regions = dataloader.dataset.normalize(
            regions_unnormalized/255,
            model.img_mean,
            model.img_std)
        regions = regions.permute(0,3,1,2)
        regions_unnormalized = regions_unnormalized.permute(0,3,1,2)
        word_idx = Variable(torch.LongTensor(data['idx']).cuda(),volatile=True)
        
        resnet_feats = model.encoder_outer(regions)
        x, x_norm = model.encoder_inner(resnet_feats)
        if exp_const.use_resnet_normalized == True:
            embedding = x_norm
        else:
            embedding = x
        
        recon_inner = model.decoder_inner(embedding)
        recon_outer = model.decoder_outer(recon_inner)
        recon_outer_gt = model.decoder_outer(resnet_feats)

        if step%10==0:
            print(exp_const.exp_name)
            
        vis_dir = os.path.join(exp_const.recon_vis_dir,str(step))
        visualize(
            255*0.5*(1+recon_outer),
            255*0.5*(1+recon_outer_gt),
            regions_unnormalized,
            vis_dir)                        


def visualize(recon,recon_gt,gt,outdir):
    io.mkdir_if_not_exists(outdir,recursive=True)
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
    torch.manual_seed(0)

    io.mkdir_if_not_exists(exp_const.recon_vis_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.encoder_outer = resnet34(pretrained=False)
    model.encoder_inner = ResnetEncoderInner(model.const.encoder_inner)
    model.decoder_outer = ResnetDecoderOuter(model.const.decoder_outer)
    model.decoder_inner = ResnetDecoderInner(model.const.decoder_inner)
    model.encoder_outer.load_state_dict(
        torch.load(model.const.encoder_outer_path))
    model.encoder_inner.load_state_dict(
        torch.load(model.const.encoder_inner_path))
    model.decoder_outer.load_state_dict(
        torch.load(model.const.decoder_outer_path))
    model.decoder_inner.load_state_dict(
        torch.load(model.const.decoder_inner_path))
    model.encoder_outer.cuda()
    model.encoder_inner.cuda()
    model.encoder_outer.eval()
    model.encoder_inner.eval()
    model.decoder_outer.cuda()
    model.decoder_inner.cuda()
    model.decoder_outer.eval()
    model.decoder_inner.eval()

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

    vis_recon(exp_const,dataloader,model)