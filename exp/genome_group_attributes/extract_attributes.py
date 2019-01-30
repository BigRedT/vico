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
from PIL import Image

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
from utils.html_writer import HtmlWriter
import utils.pytorch_layers as pytorch_layers
from .models.resnet_object import ResnetModel
from .dataset_full import GenomeAttributesDataset


def extract(model,dataloader,exp_const):
    sigmoid = nn.Sigmoid()
    #softmax = nn.Softmax()

    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean))
    img_std = Variable(torch.cuda.FloatTensor(model.img_std))

    pred_attrs = {}
    for it,data in enumerate(tqdm(dataloader)):
        # Set mode
        model.shape_net.eval()
        model.material_net.eval()
        model.color_net.eval()

        # Forward pass
        imgs = Variable(data['regions'].cuda().float()/255.,volatile=True)
        imgs = dataloader.dataset.normalize(
            imgs,
            img_mean,
            img_std)
        imgs = imgs.permute(0,3,1,2)
        cond_object_id = Variable(data['cond_object_id'].cuda(),volatile=True)
        pos_feats = Variable(data['pos_feats'].cuda(),volatile=True)
        for group in dataloader.dataset.const.groups:
            net_name = f'{group}_net'
            net = getattr(model,net_name)
            logits, last_layer_feats = net(imgs,pos_feats,cond_object_id)
            prob = sigmoid(logits)
            #prob = softmax(logits)

            prob = prob.data.cpu().numpy()
            B,D = prob.shape
            attrs = dataloader.dataset.attrs[group]
            for b in range(B):
                object_id = data['object_ids'][b]
                gt_attrs = data['attribute_synsets'][b]
                if object_id not in pred_attrs:
                    pred_attrs[object_id] = {}
                
                selected_attrs = []
                for k in range(D):
                    p = prob[b,k]
                    if p > 0.1:
                        selected_attrs.append([attrs[k],p])

                pred_attrs[object_id][group] = selected_attrs
                pred_attrs[object_id]['gt'] = gt_attrs
        

    pred_attrs_json = os.path.join(exp_const.exp_dir,'pred_attrs.json')
    io.dump_json_object(pred_attrs,pred_attrs_json)


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.vis_dir)
    save_constants(
        {'exp': exp_const,'data': data_const,'model': model_const},
        exp_const.exp_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    print('Creating shape net ...')
    model.shape_net = ResnetModel(model.const.shape_net)
    model.shape_net.load_state_dict(torch.load(model.const.shape_net_path))
    model.shape_net.cuda()
    print('Creating material net ...')
    model.material_net = ResnetModel(model.const.material_net)
    model.material_net.load_state_dict(
        torch.load(model.const.material_net_path))
    model.material_net.cuda()
    print('Creating color net ...')
    model.color_net = ResnetModel(model.const.color_net)
    model.color_net.load_state_dict(torch.load(model.const.color_net_path))
    model.color_net.cuda()
    model.img_mean = np.array([0.485, 0.456, 0.406])
    model.img_std = np.array([0.229, 0.224, 0.225])

    print('Creating dataloader ...')
    data_const = copy.deepcopy(data_const)
    dataset = GenomeAttributesDataset(data_const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets',
        'object_words','attribute_words'])
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=False,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    extract(model,dataloader,exp_const)