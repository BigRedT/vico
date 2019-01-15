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
            lr=exp_const.lr,
            weight_decay=exp_const.weight_decay)
    elif exp_const.optimizer == 'Adagrad':
        opt = optim.Adagrad(
            params,
            lr=exp_const.lr)
    else:
        assert(False), 'optimizer not implemented'

    step = 0
    if model.const.model_num is not None:
        step = model.const.model_num


    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloader):
            # Set mode
            model.net.train()

            # Forward pass
            losses = {}
            max_scores = {}
            med_scores = {}
            max_target = {}
            for cooccur_type in exp_const.cooccur_weights.keys():
                data_ = data[cooccur_type]#dataloader.dataset.get_batch_by_type(data,cooccur_type)
                # if data_ is None:
                #     losses[cooccur_type] = Variable(torch.cuda.FloatTensor([0.0]))
                #     continue

                ids1 = data_['idx1']
                ids2 = data_['idx2']
                x = Variable(torch.cuda.FloatTensor(data_['x']))
                scores = model.net(ids1,ids2,cooccur_type)
                target = torch.log(x+1e-6)
                losses[cooccur_type] = model.net.loss(scores,target,x)
                med_scores[cooccur_type] = torch.median(scores)
                max_scores[cooccur_type] = torch.max(scores)
                max_target[cooccur_type] = torch.max(target)
                # if cooccur_type=='syn' and step > 1000 and losses['syn'].data[0] > 10:
                #     B = len(ids1)
                #     loss_ = torch.pow((scores-target),2)
                #     bs = []
                #     for b_ in range(B):
                #         if loss_[b_].data[0] > 50:
                #             bs.append(b_)
                #             break

                #     ids1_ = []
                #     ids2_ = []
                #     word1_ = []
                #     word2_ = []
                #     scores_ = []
                #     targets_ = []
                #     for b in bs:
                #         ids1_.append(ids1[b])
                #         ids2_.append(ids2[b])
                #         word1_.append(dataloader.dataset.words[ids1[b]])
                #         word2_.append(dataloader.dataset.words[ids2[b]])
                #         scores_.append(scores[b].data[0])
                #         targets_.append(target[b].data[0])

                #     print(list(zip(word1_,word2_,scores_,targets_)))
            
            loss = 0
            for cooccur_type, weight in exp_const.cooccur_weights.items():
                loss = loss + weight*losses[cooccur_type]

            # Backward pass
            opt.zero_grad()
            loss.backward()
            # for p in model.net.parameters():
            #     if p.grad is not None:
            #         p.grad.data.clamp_(min=-1.0,max=1.0)
            
            opt.step()

            if step%exp_const.log_step==0:
                log_items = {'Total Loss': loss.data[0]}
                for cooccur_type,loss_ in losses.items():
                    log_items[cooccur_type + ' Loss'] = loss_.data[0]
                    log_items[cooccur_type + ' Max Score'] = \
                        max_scores[cooccur_type].data[0]
                    log_items[cooccur_type + ' Max Target'] = \
                        max_target[cooccur_type].data[0]
                    log_items[cooccur_type + ' Median Score'] = \
                        med_scores[cooccur_type].data[0]
                    log_items[cooccur_type + ' Median Target'] = \
                        np.log(dataloader.dataset.median_counts[cooccur_type])

                log_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                for name,value in log_items.items():
                    if 'Loss' in name:
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
    dataset = MultiSenseCooccurDataset(data_const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    err_msg = f'Num words mismatch (try {len(dataset.words)})'
    assert(len(dataset.words)==model.const.net.num_words), err_msg

    train_model(model,dataloader,exp_const)