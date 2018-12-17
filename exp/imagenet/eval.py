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
from collections import deque

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
import utils.pytorch_layers as pytorch_layers
from .models.resnet_normalized import ResnetNormalizedModel
from .dataset import ImagenetDataset


def accum_conf_mat(prob,labels):
    tp = prob*labels
    fp = prob*(1-labels)
    fn = (1-prob)*labels
    tn = (1-prob)*(1-labels)
    tp,tn,fp,fn = [torch.sum(k,0).view(-1,1) for k in [tp,tn,fp,fn]]
    conf_mat = torch.cat((tp,fp,fn,tn),1) # Cx4
    p = torch.sum(labels,0)
    n = torch.sum(1-labels,0)
    return conf_mat,p,n


def eval_classifiers(model,dataloader,exp_const):
    num_classes = len(dataloader.dataset.wnids)
    rep_dim = model.net.resnet_layers.fc.weight.size(1)
    
    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean),volatile=True)
    img_std = Variable(torch.cuda.FloatTensor(model.img_std),volatile=True)
    
    # Set mode
    model.net.eval()

    sig = nn.Sigmoid()

    print('Aggregating conf mat ...')
    conf_mat = np.zeros([num_classes,4])
    num_pos = np.zeros([num_classes])
    num_neg = np.zeros([num_classes])
    for it,data in enumerate(tqdm(dataloader)):
        # if it > exp_const.num_eval_batches:
        #     break
        # Forward pass
        imgs = Variable(data['img'].cuda().float()/255.,volatile=True)
        imgs = dataloader.dataset.normalize(
            imgs,
            img_mean,
            img_std)
        imgs = imgs.permute(0,3,1,2)
        logits, last_layer_feats_normalized, _ = model.net(imgs)
        prob =  sig(logits)
        conf_mat_,p,n = accum_conf_mat(
            prob.data,
            data['label_vec'].float().cuda())
        conf_mat = conf_mat + conf_mat_.cpu().numpy()
        num_pos = num_pos + p.cpu().numpy()
        num_neg = num_neg + n.cpu().numpy()
    
    conf_mat[:,0] = conf_mat[:,0] / (num_pos+1e-6)
    conf_mat[:,2] = conf_mat[:,2] / (num_pos+1e-6)
    conf_mat[:,1] = conf_mat[:,1] / (num_neg+1e-6)
    conf_mat[:,3] = conf_mat[:,3] / (num_neg+1e-6)
    conf_mat_npy = os.path.join(
        exp_const.exp_dir,
        f'conf_mat_{model.const.model_num}.npy')
    np.save(conf_mat_npy,conf_mat)

    avg_conf_mat_ = np.mean(conf_mat,0)
    avg_conf_mat = {
        'tp': avg_conf_mat_[0],
        'fp': avg_conf_mat_[1],
        'fn': avg_conf_mat_[2],
        'tn': avg_conf_mat_[3],
    }
    print(avg_conf_mat)
    avg_conf_mat_json = os.path.join(
        exp_const.exp_dir,
        f'avg_conf_mat_{model.const.model_num}.json')
    io.dump_json_object(avg_conf_mat,avg_conf_mat_json)
    

def main(exp_const,data_const,model_const):
    torch.manual_seed(0)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.net = ResnetNormalizedModel(model.const.net)
    if model.const.model_num is not None:
        model.net.load_state_dict(torch.load(model.const.net_path))
    model.net.cuda()
    model.img_mean = np.array([0.485, 0.456, 0.406])
    model.img_std = np.array([0.229, 0.224, 0.225])
    model.to_file(os.path.join(exp_const.exp_dir,'model.txt'))

    print('Creating dataloader ...')
    data_const = copy.deepcopy(data_const)
    dataset = ImagenetDataset(data_const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    print('Computing conf mat ...')
    eval_classifiers(model,dataloader,exp_const)