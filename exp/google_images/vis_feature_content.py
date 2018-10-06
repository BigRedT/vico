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


def smoothness(x):
    B,H,W,C = x.size()
    r1 = x[:,:H-1,:,:]
    r2 = x[:,1:H,:,:]
    h1 = x[:,:,:W-1,:]
    h2 = x[:,:,1:W,:]
    loss = torch.mean((r1-r2)*(r1-r2)) + \
         torch.mean((h1-h2)*(h1-h2))
    return loss


def generate_images(exp_const,dataloader,model):
    count = 0
    criterion = nn.MSELoss()
    sig = nn.Sigmoid()
    for it,data in enumerate(dataloader):
        print(f'Iter: {it}')
        if count>=exp_const.num_gen_imgs:
            break
        x_unnormalized = Variable(
            torch.FloatTensor(data['img']).cuda(),
            requires_grad=False)
        noise = torch.FloatTensor(255*np.random.uniform(size=data['img'].shape))
        y_logits = Variable(
            -torch.log(255/torch.FloatTensor(0.5*data['img']+0.5*noise).cuda()-1),
            requires_grad=True)
        
        opt = optim.SGD(
            [y_logits],
            lr=exp_const.lr,
            momentum=exp_const.momentum)
        # opt = optim.Adam(
        #     [y_logits],
        #     lr=exp_const.lr)
        
        x = dataloader.dataset.normalize(
            x_unnormalized/255,
            model.img_mean,
            model.img_std)
        x = x.permute(0,3,1,2)
        
        if exp_const.use_resnet_normalized == True:
            _, x_feat, _ = model.net(x)
        else:
            _, x_feat = model.net(x)
            
        for k in range(exp_const.max_inner_iter):
            y = dataloader.dataset.normalize(
                sig(y_logits),
                model.img_mean,
                model.img_std)
            y = y.permute(0,3,1,2)

            if exp_const.use_resnet_normalized == True:
                _, y_feat, _ = model.net(y)
            else:
                _, y_feat = model.net(y)

            net_loss = criterion(
                y_feat,
                Variable(x_feat.data,requires_grad=False))
            smoothness_loss = 1e-2*smoothness(sig(y_logits))
            loss = net_loss + smoothness_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if k%10==0:
                print_str = \
                    'Iter: {} | ' + \
                    'Total Loss: {} | ' + \
                    'Net Loss: {} | ' + \
                    'Smoothness Loss: {}'
                print_str = print_str.format(
                    k,
                    loss.data[0],
                    net_loss.data[0],
                    smoothness_loss.data[0])
                print(print_str)

        y_unnormalized = 255*sig(y_logits)
        visualize(y_unnormalized,x_unnormalized,exp_const.deepdream_dir,it)
        count += y_logits.size(0)


def visualize(recon,gt,outdir,it):
    recon = recon.data.cpu().numpy()
    gt = gt.data.cpu().numpy()
    html_filename = os.path.join(outdir,f'deepdream_{it}.html')
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
    io.mkdir_if_not_exists(exp_const.deepdream_dir)

    torch.manual_seed(0)

    print('Creating network ...')
    model = Model()
    model.const = model_const
    if exp_const.use_resnet_normalized == True:
        model.net = resnet152_normalized(pretrained=False)
    else:
        model.net = resnet152(pretrained=False)
    model.net.load_state_dict(torch.load(model.const.net_path))
    model.net.cuda()
    model.net.eval()
    model.img_mean = Variable(
        torch.FloatTensor(np.array([0.485, 0.456, 0.406]))).cuda()
    model.img_std = Variable(
        torch.FloatTensor(np.array([0.229, 0.224, 0.225]))).cuda()

    print('Creating dataloader ...')
    dataset = GoogleImagesImageLevelDataset(data_const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    generate_images(exp_const,dataloader,model)