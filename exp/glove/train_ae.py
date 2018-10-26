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
from exp.glove.concat_embed_dataset import ConcatEmbedDataset
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
    glove_dim = dataloader.dataset.const.glove_dim
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

            concat_embeddings = Variable(data['embedding']).cuda()
            x, embedding = model.encoder(concat_embeddings)
            recon_concat_embeddings = model.decoder(embedding)
            
            recon_loss = model.decoder.compute_loss(
                recon_concat_embeddings,
                concat_embeddings)

            glove = normalize(concat_embeddings[:,:glove_dim])
            visual = normalize(concat_embeddings[:,glove_dim:])
            glove_dot = torch.mm(glove,torch.transpose(glove,0,1))
            visual_dot = torch.mm(visual,torch.transpose(visual,0,1))
            embedding_dot = torch.mm(embedding,torch.transpose(embedding,0,1))
            max_dot = torch.max(glove_dot,visual_dot)
            min_dot = torch.min(glove_dot,visual_dot)
            dot_recon_loss = 0*(
                torch.mean(torch.max(0*min_dot,min_dot.detach()-embedding_dot)) + \
                torch.mean(torch.max(0*min_dot,embedding_dot-max_dot.detach())))
            # mean_dot = 0.5*(glove_dot + visual_dot)
            # dot_recon_loss = model.decoder.compute_loss(
            #     embedding_dot,mean_dot.detach())
            
            l2_loss = 1e-3*model.decoder.compute_loss(x,0*x)
            total_loss = recon_loss + dot_recon_loss + l2_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            if step%10==0:
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    'Total Loss: {:.4f} | ' + \
                    'Recon Loss: {:.4f} | ' + \
                    'Dot Recon Loss: {:.4f} | ' + \
                    'L2 Loss: {:.4f} | '
                log_str = log_str.format(
                    epoch,
                    it,
                    step,
                    total_loss.data[0],
                    recon_loss.data[0],
                    dot_recon_loss.data[0],
                    l2_loss.data[0])
                print(log_str)
                log_value('Total Loss',total_loss.data[0],step)
                log_value('Recon Loss',recon_loss.data[0],step)
                log_value('Dot Recon Loss',dot_recon_loss.data[0],step)
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
    dataset = ConcatEmbedDataset(data_const)
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True)

    train_model(model,dataloader,exp_const)