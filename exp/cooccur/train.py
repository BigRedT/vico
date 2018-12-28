import os
import h5py
import math
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from tensorboard_logger import configure, log_value
import numpy as np

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
import utils.pytorch_layers as pytorch_layers
from .models.logbilinear import LogBilinear, LogBilinearNegPairs
from .dataset import CooccurDataset


def train_model(model,dataloader,exp_const):
    params = model.net.parameters()
    
    if exp_const.optimizer == 'SGD':
        opt = optim.SGD(
            params,
            lr=exp_const.lr,
            momentum=exp_const.momentum)
    elif exp_const.optimizer == 'Adam':
        opt = optim.Adam(
            params,
            lr=exp_const.lr)
    elif exp_const.optimizer == 'Adagrad':
        opt = optim.Adagrad(
            params,
            lr=exp_const.lr)
    else:
        assert(False), 'optimizer not implemented'

    step = 0
    if model.const.model_num is not None:
        step = model.const.model_num

    print('Creating cooccur matrix')
    cooccur = dataloader.dataset.cooccur
    words = dataloader.dataset.words
    word_to_idx = dataloader.dataset.word_to_idx
    num_words = len(cooccur)
    cooccur_mat = np.zeros([num_words,num_words],dtype=np.float32)
    for i,word1 in enumerate(tqdm(words)):
        context = cooccur[word1]
        for word2,count in context.items():
            j = word_to_idx[word2]
            cooccur_mat[i,j] = count
    cooccur_mat = Variable(torch.cuda.FloatTensor(cooccur_mat))

    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloader):
            # Set mode
            model.net.train()

            # Forward pass
            ids1 = data['idx1']
            ids2 = data['idx2']
            scores = model.net(ids1,ids2)

            # Compute loss
            x = Variable(torch.cuda.FloatTensor(data['x']))
            loss = model.net.loss(scores,x)
            # loss = model.net.loss(
            #     scores,
            #     ids1,
            #     ids2,
            #     cooccur_mat)
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                log_items = {
                    'Loss': loss.data[0],
                }

                log_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                for name,value in log_items.items():
                    log_str += '{}: {:.4f} | '.format(name,value)
                    log_value(name,value,step)

                print(log_str)
            
            if step%(100*exp_const.log_step)==0:
                print(f'Experiment: {exp_const.exp_name}')
                
            if step%exp_const.model_save_step==0:
                save_items = {
                    'net': model.net,
                }

                for name,nn_model in save_items.items():
                    model_path = os.path.join(
                        exp_const.model_dir,
                        f'{name}_{step}')
                    torch.save(nn_model.state_dict(),model_path)

            step += 1


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    io.mkdir_if_not_exists(exp_const.vis_dir)
    configure(exp_const.log_dir)
    save_constants(
        {'exp': exp_const,'data': data_const,'model': model_const},
        exp_const.exp_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.net = LogBilinear(model.const.net)
    if model.const.model_num is not None:
        model.net.load_state_dict(torch.load(model.const.net_path))
    model.net.cuda()
    model.to_file(os.path.join(exp_const.exp_dir,'model.txt'))

    print('Creating dataloader ...')
    dataset = CooccurDataset(data_const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    train_model(model,dataloader,exp_const)