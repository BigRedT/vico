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
from .models.logbilinear import LogBilinear
from .dataset import MultiSenseCooccurDataset
from .neg_dataset import NegMultiSenseCooccurDataset


def train_model(model,dataloader,neg_dataloader,exp_const):
    params = model.net.parameters()
    
    if exp_const.optimizer == 'SGD':
        opt = optim.SGD(
            params,
            lr=exp_const.lr,
            momentum=exp_const.momentum,
            weight_decay=exp_const.weight_decay)
    elif exp_const.optimizer == 'Adam':
        opt = optim.Adam(
            params,
            lr=exp_const.lr,
            weight_decay=exp_const.weight_decay)
    elif exp_const.optimizer == 'Adagrad':
        opt = optim.Adagrad(
            params,
            lr=exp_const.lr,
            weight_decay=exp_const.weight_decay)
    else:
        assert(False), 'optimizer not implemented'

    step = 0
    if model.const.model_num is not None:
        step = model.const.model_num

    neg_it = iter(neg_dataloader)
    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloader):
            
            # Get negative samples
            neg_data = next(neg_it,None)
            if neg_data is None:
                neg_it = iter(neg_dataloader)
                neg_data = next(neg_it,None)

            # Set mode
            model.net.train()

            # Forward pass
            losses = {}
            neg_losses = {}
            for cooccur_type in exp_const.cooccur_weights.keys():
                # Positives
                data_ = data[cooccur_type]
                ids1 = data_['idx1']
                ids2 = data_['idx2']
                x = Variable(torch.cuda.FloatTensor(data_['x']))
                scores = model.net(ids1,ids2,cooccur_type)
                target = torch.log(x+1e-6)
                losses[cooccur_type] = model.net.loss(scores,target,x)
                
                # Negatives
                data_ = neg_data[cooccur_type]
                ids1 = data_['idx1']
                ids2 = data_['idx2']
                x = Variable(torch.cuda.FloatTensor(data_['x']))
                scores = model.net(ids1,ids2,cooccur_type)
                losses[f'Neg {cooccur_type}'] = torch.mean(torch.max(0*scores,scores))

            loss = 0
            for cooccur_type, weight in exp_const.cooccur_weights.items():
                loss = loss + \
                    weight*losses[cooccur_type] + \
                    exp_const.use_neg*weight*losses[f'Neg {cooccur_type}']

            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                log_items = {'Total Loss': loss.data[0]}
                for cooccur_type,loss_ in losses.items():
                    log_items[cooccur_type + ' Loss'] = loss_.data[0]

                log_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                for name,value in log_items.items():
                    log_value(name,value,step)
                    if 'Loss' in name and 'Neg' not in name:
                        log_str += '{}: {:.4f} | '.format(name,value)

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
    if model_const.model_num is None:
        const_dict = {
            'exp': exp_const,
            'data': data_const,
            'model': model_const}
    else:
        const_dict = {
            f'exp_finetune_{model_const.model_num}': exp_const,
            f'data_finetune_{model_const.model_num}': data_const,
            f'model_finetune_{model_const.model_num}': model_const}
    save_constants(
        const_dict,
        exp_const.exp_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.net = LogBilinear(model.const.net)
    if model.const.model_num is not None:
        model.net.load_state_dict(torch.load(model.const.net_path))
    model.net.cuda()
    model.to_file(os.path.join(exp_const.exp_dir,'model.txt'))

    print('Creating positive dataloader ...')
    dataset = MultiSenseCooccurDataset(data_const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    print('Creating negative dataloader ...')
    neg_dataset = NegMultiSenseCooccurDataset(data_const)
    collate_fn = neg_dataset.create_collate_fn()
    neg_dataloader = DataLoader(
        neg_dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    err_msg = f'Num words mismatch (try {len(dataset.words)})'
    assert(len(dataset.words)==model.const.net.num_words), err_msg

    train_model(model,dataloader,neg_dataloader,exp_const)