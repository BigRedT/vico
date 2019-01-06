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
from .models.resnet import ResnetModel
from .models.embed_to_classifier import Embed2Class
from .dataset import Cifar100Dataset


def eval_model(model,dataloader,exp_const):
    # Set mode
    model.net.eval()
    if exp_const.feedforward==False:
        model.embed2class.eval()

    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean))
    img_std = Variable(torch.cuda.FloatTensor(model.img_std))

    criterion = nn.CrossEntropyLoss()

    avg_loss = 0
    correct = 0
    num_samples = 0
    num_classes = len(dataloader.dataset.labels)
    confmat = np.zeros([num_classes,num_classes])
    for it,data in enumerate(tqdm(dataloader)):
        # Forward pass
        imgs = Variable(data['img'].cuda().float()/255)
        imgs = dataloader.dataset.normalize(
            imgs,
            img_mean,
            img_std)
        imgs = imgs.permute(0,3,1,2)
        label_idxs = Variable(data['label_idx'].cuda())

        if exp_const.feedforward==True:
            logits, feats = model.net(imgs)
        else:
            _, feats = model.net(imgs)
            class_weights = model.embed2class()
            logits = model.embed2class.classify(feats,class_weights)

        # Computer loss
        loss = criterion(logits,label_idxs)    

        _,argmax = torch.max(logits,1)
        argmax = argmax.data.cpu().numpy()
        label_idxs_ = label_idxs.data.cpu().numpy()

        # Aggregate loss or accuracy
        batch_size = imgs.size(0)
        num_samples += batch_size
        avg_loss += (loss.data[0]*batch_size)
        correct += np.sum(argmax==label_idxs_)

        for b in range(argmax.shape[0]):
            gt_idx = label_idxs_[b]
            pred_idx = argmax[b]
            confmat[gt_idx,pred_idx] += 1

    avg_loss = avg_loss / num_samples
    acc = correct / float(num_samples)

    eval_results = {
        'Avg Loss': avg_loss, 
        'Acc': acc,
        'Conf Mat': confmat
    }

    return eval_results


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.vis_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    
    model.net = ResnetModel(model.const.net)
    if model.const.model_num is not None:
        model.net.load_state_dict(torch.load(model.const.net_path))
    model.net.cuda()
    
    if exp_const.feedforward==False:
        model.embed2class = Embed2Class(model.const.embed2class)
        if model.const.model_num is not None:
            model.embed2class.load_state_dict(
                torch.load(model.const.embed2class_path))
        model.embed2class.cuda()
    
    model.img_mean = np.array([0.485, 0.456, 0.406])
    model.img_std = np.array([0.229, 0.224, 0.225])

    print('Creating dataloader ...')
    dataset = Cifar100Dataset(data_const)
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers)

    eval_results = eval_model(model,dataloader,exp_const)

    confmat_npy = os.path.join(exp_const.exp_dir,'confmat.npy')
    np.save(confmat_npy,eval_results['Conf Mat'])

    results = {
        'Avg Loss': eval_results['Avg Loss'],
        'Acc': eval_results['Acc']
    }

    print(results)
    results_json = os.path.join(exp_const.exp_dir,'results.json')
    io.dump_json_object(results,results_json)

    embeddings_npy = os.path.join(exp_const.exp_dir,'embeddings.npy')
    if exp_const.feedforward==True:
        np.save(
            embeddings_npy,
            model.net.resnet_layers.fc.weight.data.cpu().numpy())
    else:
        np.save(
            embeddings_npy,
            model.embed2class.embed.weight.data.cpu().numpy())

    labels_npy = os.path.join(exp_const.exp_dir,'labels.npy')
    np.save(labels_npy,dataset.labels)