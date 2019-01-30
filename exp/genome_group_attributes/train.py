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
from .dataset import GenomeAttributesDataset


def train_model(model,dataloader,exp_const):
    params = model.net.parameters()
    
    lr = exp_const.lr
    if exp_const.optimizer == 'SGD':
        opt = optim.SGD(
            params,
            lr=lr,
            momentum=exp_const.momentum)
    elif exp_const.optimizer == 'Adam':
        opt = optim.Adam(
            params,
            lr=lr)
    else:
        assert(False), 'optimizer not implemented'

    bce_criterion = nn.BCELoss()
    
    sigmoid = nn.Sigmoid()
    #softmax = nn.Softmax()

    step = 0
    if model.const.model_num is not None:
        step = model.const.model_num

    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean))
    img_std = Variable(torch.cuda.FloatTensor(model.img_std))

    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloader):
            lr = exp_const.lr*0.5**(step//2000)
            pytorch_layers.set_learning_rate(opt,lr)

            # Set mode
            model.net.train()

            # Forward pass
            imgs = Variable(data['region'].cuda().float()/255.)
            imgs = dataloader.dataset.normalize(
                imgs,
                img_mean,
                img_std)
            imgs = imgs.permute(0,3,1,2)
            obj_ids = Variable(data['object_id'].cuda())
            pos_feats = Variable(data['pos_feat'].cuda())
            logits, last_layer_feats = model.net(imgs,pos_feats,obj_ids)
            prob = sigmoid(logits)
            #prob = softmax(logits)
            
            prob_ = prob.data.cpu().numpy()
            B,C = prob_.shape
            gt_0_3 = np.sum(prob_>0.3) / (B*C)
            gt_0_5 = np.sum(prob_>0.5) / (B*C)
            gt_0_7 = np.sum(prob_>0.7) / (B*C)
            gt_0_9 = np.sum(prob_>0.9) / (B*C)

            # Computer loss
            attr_labels = Variable(data['label_vec'].cuda())
            bce_loss = bce_criterion(prob,attr_labels)
            loss = bce_loss

            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                log_items = {
                    'Loss': loss.data[0],
                    'BCE Loss': bce_loss.data[0],
                    'LR': lr,
                    'gt_0.3': gt_0_3,
                    'gt_0.5': gt_0_5,
                    'gt_0.7': gt_0_7,
                    'gt_0.9': gt_0_9,
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

            if step%exp_const.vis_step==0:
                top_prob, top_ids = torch.topk(prob,5,1)
                top_prob = top_prob.data.cpu().numpy()
                top_ids = top_ids.data.cpu().numpy()
                top_attrs = []
                recall = 0
                gt_attrs = data['attributes']
                for b in range(top_ids.shape[0]):
                    pred_attrs = [
                        dataloader.dataset.attrs[idx] for idx in top_ids[b]]
                    top_attrs.append(pred_attrs)
                    gt_attrs_set = set(gt_attrs[b])
                    pred_attrs_set = set(pred_attrs)
                    recall += len(gt_attrs_set.intersection(pred_attrs_set)) / \
                        len(gt_attrs_set)

                recall = recall / (top_ids.shape[0] + 1e-6)
                log_value('Recall',recall,step)
                print('Recall', recall)

                vis_pred(
                    data['region'].numpy(),
                    gt_attrs,
                    top_prob,
                    top_attrs,
                    step,
                    exp_const)

            step += 1


def vis_pred(imgs,gt_attrs,top_prob,top_attrs,step,exp_const):
    vis_dir = os.path.join(exp_const.vis_dir,str(step))
    io.mkdir_if_not_exists(vis_dir,recursive=True)
    html_filename = os.path.join(vis_dir,'vis.html')
    html_writer = HtmlWriter(html_filename)
    col_dict = {
        0: 'Img',
        1: 'GT Attributes',
        2: 'Pred Attribtes'
    }
    html_writer.add_element(col_dict)
    B = imgs.shape[0]
    for b in range(B):
        img = Image.fromarray(imgs[b])
        img_name = str(b) + '.png'
        img_path = os.path.join(vis_dir,img_name)
        img.save(img_path)
        
        pred_str = ''
        for attr, p in zip(top_attrs[b],top_prob[b]):
            pred_str += attr + '(' + str(round(p,3)) + ')' + '&nbsp;'*4
        
        col_dict = {
            0: html_writer.image_tag(img_name),
            1: gt_attrs[b],
            2: pred_str
        }
        html_writer.add_element(col_dict)
    
    html_writer.close()

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
    model.net = ResnetModel(model.const.net)
    if model.const.model_num is not None:
        model.net.load_state_dict(torch.load(model.const.net_path))
    model.net.cuda()
    model.img_mean = np.array([0.485, 0.456, 0.406])
    model.img_std = np.array([0.229, 0.224, 0.225])
    model.to_file(os.path.join(exp_const.exp_dir,'model.txt'))

    print('Creating dataloader ...')
    data_const = copy.deepcopy(data_const)
    dataset = GenomeAttributesDataset(data_const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    train_model(model,dataloader,exp_const)