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

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
import utils.pytorch_layers as pytorch_layers
from exp.visualgenome.image_regions_dataset import ImageRegionsDataset
from exp.visualgenome.models.resnet import resnet152
from exp.visualgenome.models.resnet_normalized import resnet152_normalized
from exp.visualgenome.models.word_classification_mlp import WordClassifierLayer


def l2_norm_sq(x):
    return torch.mean(torch.sum(x*x,1))


def train_model(exp_const,dataloader,model):
    
    if exp_const.train_net==True:
        params = itertools.chain(        
            model.object_classifier_layers.parameters(),
            model.attribute_classifier_layers.parameters(),
            model.net.parameters())
    else:
        params = itertools.chain(        
            model.object_classifier_layers.parameters(),
            model.attribute_classifier_layers.parameters())

    if exp_const.optimizer == 'SGD':
        opt = optim.SGD(
            params,
            lr=exp_const.lr,
            momentum=exp_const.momentum)
    elif exp_const.optimizer == 'Adam':
        opt = optim.Adam(
            params,
            lr=exp_const.lr)
    else:
        assert(False), 'optimizer not implemented'
    
    object_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    attribute_criterion = nn.BCEWithLogitsLoss()
    sigmoid = pytorch_layers.get_activation('Sigmoid')
    if exp_const.load_finetuned is True:
        step = model.const.model_num
    else:
        step = -1
    for epoch in range(exp_const.num_epochs):
        k = -1
        for data in dataloader:
            # if k >= 1:
            #     break
            if data is None:
                continue
            
            regions = dataloader.dataset.normalize(
                data['regions']/255,
                model.img_mean,
                model.img_std)
            regions = Variable(torch.FloatTensor(regions).cuda())
            regions = regions.permute(0,3,1,2)
            object_labels_idx = Variable(torch.LongTensor(
                data['object_labels_idx']).cuda())
            attribute_labels = Variable(torch.FloatTensor(
                data['attribute_labels']).cuda())
            B = regions.size(0)
            rB = exp_const.region_batch_size
            for i in range(math.ceil(B/rB)):
                step += 1
                k += 1
                r = min(i*rB+rB,B)
                if exp_const.use_resnet_normalized == True:
                    out, last_layer_features, last_layer_features_ = \
                        model.net(regions[i*rB:r])
                else:
                    out, last_layer_features = model.net(regions[i*rB:r])
                object_scores = model.object_classifier_layers(
                    last_layer_features)
                attribute_scores = model.attribute_classifier_layers(
                    last_layer_features)
                object_loss = object_criterion(
                    object_scores,
                    object_labels_idx[i*rB:r])
                attribute_loss = attribute_criterion(
                    attribute_scores,
                    attribute_labels[i*rB:r])
                feat_l2_norm_sq = l2_norm_sq(last_layer_features_)
                loss = object_loss + attribute_loss
                if exp_const.use_resnet_normalized == True:
                    loss = loss + (1e-6)*feat_l2_norm_sq
                opt.zero_grad()
                loss.backward()
                for p in params:
                    p.grad = torch.clamp(p.grad,-0.1,0.1)
                opt.step()
            
                if step%10==0:
                    log_str = \
                        'Epoch: {} | Iter: {} | Step: {} | ' + \
                        'Total Loss: {:.4f} | ' + \
                        'Object Loss: {:.4f} | ' + \
                        'Attribute Loss: {:.4f} | ' + \
                        'Feat L2: {:.4f}'
                    log_str = log_str.format(
                        epoch,
                        k,
                        step,
                        loss.data[0],
                        object_loss.data[0],
                        attribute_loss.data[0],
                        feat_l2_norm_sq.data[0])
                    print(log_str)
                    log_value('Total Loss',loss.data[0],step)
                    log_value('Object Loss',object_loss.data[0],step)
                    log_value('Attribute Loss',attribute_loss.data[0],step)
                    log_value('Feat L2 Sq Loss',feat_l2_norm_sq.data[0],step)
                
                if step%1000==0:
                    net_path = os.path.join(
                        exp_const.model_dir,
                        f'net_{step}')
                    object_classifier_layers_path = os.path.join(
                        exp_const.model_dir,
                        f'object_classifier_layers_{step}')
                    attribute_classifier_layers_path = os.path.join(
                        exp_const.model_dir,
                        f'attribute_classifier_layers_{step}')
                    torch.save(model.net.state_dict(),net_path)
                    torch.save(
                        model.object_classifier_layers.state_dict(),
                        object_classifier_layers_path)
                    torch.save(
                        model.attribute_classifier_layers.state_dict(),
                        attribute_classifier_layers_path)



def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    configure(exp_const.log_dir)
    save_constants({'exp': exp_const,'data': data_const},exp_const.exp_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    if exp_const.use_resnet_normalized == True:
        model.net = resnet152_normalized(pretrained=True)
    else:
        model.net = resnet152(pretrained=True)
    model.object_classifier_layers = WordClassifierLayer(
        model_const.object_classifier_layers)
    model.attribute_classifier_layers = WordClassifierLayer(
        model_const.attribute_classifier_layers)
    if exp_const.load_finetuned == True:
        model.net.load_state_dict(torch.load(model.const.net_path))
        model.object_classifier_layers.load_state_dict(torch.load(
            model.const.object_classifier_layers_path))
        model.attribute_classifier_layers.load_state_dict(torch.load(
            model.const.attribute_classifier_layers_path))
    model.net.cuda()
    model.object_classifier_layers.cuda()
    model.attribute_classifier_layers.cuda()
    model.net.train()
    model.object_classifier_layers.train()
    model.attribute_classifier_layers.train()
    model.img_mean = np.array([0.485, 0.456, 0.406])
    model.img_std = np.array([0.229, 0.224, 0.225])
    model.to_file(os.path.join(exp_const.exp_dir,'model.txt'))

    print('Creating dataloader ...')
    dataset = ImageRegionsDataset(data_const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets','object_ids'])
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    train_model(exp_const,dataloader,model)