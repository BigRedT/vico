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
from .models.resnet import resnet50
from .dataset import GenomeAttributesDataset


def extract_feats(model,dataloader,exp_const):
    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean),volatile=True)
    img_std = Variable(torch.cuda.FloatTensor(model.img_std),volatile=True)
    
    # Set mode
    model.net.eval()

    feats = []
    object_ids = []
    image_ids = []
    for it,data in enumerate(tqdm(dataloader)):
        if data is None:
            continue
    
        imgs = Variable(data['regions'].cuda().float()/255.,volatile=True)
        imgs = dataloader.dataset.normalize(
            imgs,
            img_mean,
            img_std)
        imgs = imgs.permute(0,3,1,2)
        
        x = model.net(imgs)
        x = x.data.cpu().numpy()
        
        feats.append(x)
        object_ids += data['object_ids'].tolist()
        image_ids += data['image_ids'].tolist()

    feats = np.concatenate(feats)

    print('Saving feats and ids ...')
    np.save(os.path.join(exp_const.exp_dir,'feats.npy'),feats)
    io.dump_json_object(
        object_ids,
        os.path.join(exp_const.exp_dir,'object_ids.json'))
    io.dump_json_object(
        image_ids,
        os.path.join(exp_const.exp_dir,'image_ids.json'))


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    save_constants(
        {'exp': exp_const,'data': data_const,'model': model_const},
        exp_const.exp_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.net = resnet50()
    state_dict = torch.load(model.const.net_path)
    
    model.net.load_state_dict(state_dict)
    model.net.cuda()
    model.img_mean = np.array([0.485, 0.456, 0.406])
    model.img_std = np.array([0.229, 0.224, 0.225])

    print('Creating dataloader ...')
    data_const = copy.deepcopy(data_const)
    dataset = GenomeAttributesDataset(data_const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets','attribute_labels_idxs',
        'object_words','attribute_words'])
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    print('Begin feature extraction ...')
    extract_feats(model,dataloader,exp_const)