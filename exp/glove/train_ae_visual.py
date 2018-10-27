import os
import h5py
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
from exp.glove.visual_features_dataset import VisualFeaturesDataset
from exp.glove.models.encoder import Encoder
from exp.glove.models.decoder import Decoder
from utils.pytorch_layers import adjust_learning_rate


def compute_norm(x):
    return torch.norm(x,p=2,dim=1,keepdim=True)

def normalize(x):
    return x / (1e-6 + compute_norm(x))


def train_model(model,dataloader,exp_const):
    params = itertools.chain(
        model.encoder.parameters(),
        model.decoder.parameters())
    opt = optim.Adam(params,lr=exp_const.lr)
    step = -1
    for epoch in range(exp_const.num_epochs):
        # adjust_learning_rate(
        #     opt,
        #     exp_const.lr,
        #     epoch,
        #     decay_by=0.5,
        #     decay_every=10)
        for it, data in enumerate(dataloader):
            step += 1

            model.encoder.train()
            model.decoder.train()

            features = Variable(data['feature']).cuda()
            x, ae_features = model.encoder(features)
            recon_features = model.decoder(ae_features)
            
            recon_loss = model.decoder.compute_loss(
                recon_features,
                features)
            
            l2_loss = 1e-3*model.decoder.compute_loss(x,0*x)
            total_loss = recon_loss + l2_loss

            mean_feature = torch.mean(features,dim=0,keepdim=True)
            mean_loss = model.decoder.compute_loss(
                mean_feature.repeat(features.size(0),1),
                features)

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            if step%10==0:
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    'Total Loss: {:.6f} | ' + \
                    'Recon Loss: {:.6f} | ' + \
                    'L2 Loss: {:.6f} | ' + \
                    'Mean Loss: {:.6f} |'
                log_str = log_str.format(
                    epoch,
                    it,
                    step,
                    total_loss.data[0],
                    recon_loss.data[0],
                    l2_loss.data[0],
                    mean_loss.data[0])
                print(log_str)
                log_value('Total Loss',total_loss.data[0],step)
                log_value('Recon Loss',recon_loss.data[0],step)
                log_value('L2 Loss',l2_loss.data[0],step)

            if step%100==0:
                print(exp_const.exp_name)

        if epoch%10==0:
            encoder_path = os.path.join(
                exp_const.model_dir,
                f'encoder_{epoch}')
            torch.save(model.encoder.state_dict(),encoder_path)

            decoder_path = os.path.join(
                exp_const.model_dir,
                f'decoder_{epoch}')
            torch.save(model.decoder.state_dict(),decoder_path)
        


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    configure(exp_const.log_dir)
    save_constants({
        'exp': exp_const,
        'data': data_const,
        'model': model_const},
        exp_const.exp_dir)

    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.encoder = Encoder(model.const.encoder).cuda()
    model.decoder = Decoder(model.const.decoder).cuda()

    encoder_path = os.path.join(
        exp_const.model_dir,
        f'encoder_{-1}')
    torch.save(model.encoder.state_dict(),encoder_path)

    decoder_path = os.path.join(
        exp_const.model_dir,
        f'decoder_{-1}')
    torch.save(model.decoder.state_dict(),decoder_path)

    print('Creating dataloader ...')
    dataset = VisualFeaturesDataset(data_const)
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True)

    train_model(model,dataloader,exp_const)