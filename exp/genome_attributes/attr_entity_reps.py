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
from exp.imagenet.models.resnet_normalized import ResnetNormalizedModel
from .dataset import GenomeAttributesDataset


def classifier_attr_entity_reps(model,exp_const):
    classifier_reps = model.net.resnet_layers.fc.weight.data.cpu().numpy()
    classifier_reps = classifier_reps / \
        (np.linalg.norm(classifier_reps,ord=2,axis=1,keepdims=True)+1e-6)
    np.save(
        os.path.join(exp_const.exp_dir,'classifier_reps.npy'),
        classifier_reps)


def compute_attr_entity_reps(model,dataloader,exp_const):
    num_classes = len(dataloader.dataset.sorted_attribute_synsets)
    rep_dim = model.net.resnet_layers.fc.weight.size(1)

    print('Allocating memory for storing attr-entity reps ...')
    reps = np.zeros([num_classes,rep_dim],dtype=np.float32)
    num_imgs_per_class = np.zeros([num_classes],dtype=np.int32)
    
    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean),volatile=True)
    img_std = Variable(torch.cuda.FloatTensor(model.img_std),volatile=True)
    
    # Set mode
    model.net.eval()

    print('Aggregating image features ...')
    for it,data in enumerate(tqdm(dataloader)):
        # Forward pass
        imgs = Variable(data['regions'].cuda().float()/255.,volatile=True)
        imgs = dataloader.dataset.normalize(
            imgs,
            img_mean,
            img_std)
        imgs = imgs.permute(0,3,1,2)
        last_layer_feats_normalized, _ = model.net.forward_features_only(imgs)
    
        last_layer_feats_normalized = \
            last_layer_feats_normalized.data.cpu().numpy()
        for b in range(last_layer_feats_normalized.shape[0]):
            on_attrs = data['attribute_synsets'][b]
            for attr in on_attrs:
                i = dataloader.dataset.attribute_synset_to_idx[attr]
                num_imgs_per_class[i] = num_imgs_per_class[i] + 1
                reps[i] = reps[i] + last_layer_feats_normalized[b]

    reps = reps / (num_imgs_per_class[:,None]+1e-6)
    reps = reps / (np.linalg.norm(reps,ord=2,axis=1,keepdims=True)+1e-6)

    print('Saving attr-entity reps and related files ...')
    np.save(os.path.join(exp_const.exp_dir,'reps.npy'),reps)
    np.save(
        os.path.join(exp_const.exp_dir,'num_imgs_per_class.npy'),
        num_imgs_per_class)
    io.dump_json_object(
        dataloader.dataset.attribute_synset_to_idx,
        os.path.join(exp_const.exp_dir,'wnid_to_idx.json'))


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    save_constants(
        {'exp': exp_const,'data': data_const,'model': model_const},
        exp_const.exp_dir)
    
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
    dataset = GenomeAttributesDataset(data_const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets','attribute_labels_idxs'])
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    print('Mean Image Features Based attr-entity Reps ...')
    compute_attr_entity_reps(model,dataloader,exp_const)

    # print('Classifier Based Attr-Attr Reps ...')
    # classifier_attr_entity_reps(model,exp_const)