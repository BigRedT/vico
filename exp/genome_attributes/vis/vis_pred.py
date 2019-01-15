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
import skimage.io as skio

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
from utils.html_writer import HtmlWriter
import utils.pytorch_layers as pytorch_layers
from ..models.resnet import ResnetModel
from ..dataset import GenomeAttributesDataset


def eval_model(model,dataloader,exp_const):
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax()

    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean))
    img_std = Variable(torch.cuda.FloatTensor(model.img_std))

    model.net.eval()

    vis_dir = os.path.join(exp_const.vis_dir,'top_preds')
    io.mkdir_if_not_exists(vis_dir)

    filename = os.path.join(vis_dir,'vis.html')
    html_writer = HtmlWriter(filename)
    
    for it,data in enumerate(dataloader):
        if it >= 10:
            break

        # Forward pass
        imgs = Variable(data['regions'].cuda().float()/255.)
        imgs = dataloader.dataset.normalize(
            imgs,
            img_mean,
            img_std)
        imgs = imgs.permute(0,3,1,2)
        logits, last_layer_feats = model.net(imgs)
        #prob = sigmoid(logits)
        prob = softmax(logits)
        
        prob_ = prob.data.cpu().numpy()
        B,C = prob_.shape
        gt_0_3 = np.sum(prob_>0.3) / (B*C)
        gt_0_5 = np.sum(prob_>0.5) / (B*C)
        gt_0_7 = np.sum(prob_>0.7) / (B*C)
        gt_0_9 = np.sum(prob_>0.9) / (B*C)

        print(gt_0_3,gt_0_5,gt_0_7,gt_0_9)

        attr_labels_idxs = data['attribute_labels_idxs']
    
        top_probs,top_ids = torch.topk(prob,exp_const.num_nbrs)
        top_probs = top_probs.data.cpu().numpy()
        top_ids = top_ids.data.cpu().numpy()
        B = top_ids.shape[0]

        imgs_ = data['regions'].numpy()
        for b in range(B):
            top_ids_ = top_ids[b]
            top_probs_ = top_probs[b]
            pred_str = ''
            for k in range(exp_const.num_nbrs):
                idx = top_ids_[k]
                pred_str += dataloader.dataset.sorted_attribute_synsets[idx]
                score = str(round(top_probs_[k],2))
                pred_str += f' ({score})'
                pred_str += '&nbsp;'*2

            img_ = imgs_[b]
            img_name = f'{it}_{b}.png'
            img_png = os.path.join(vis_dir,img_name)
            skio.imsave(img_png,img_)
            col_dict = {
                0: html_writer.image_tag(img_name),
                1: data['attribute_synsets'][b],
                2: pred_str
            }

            html_writer.add_element(col_dict)

    html_writer.close()

    
def main(exp_const,data_const,model_const):
    torch.manual_seed(0)

    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.net = ResnetModel(model.const.net)
    if model.const.model_num is not None:
        model.net.load_state_dict(torch.load(model.const.net_path))

    model.net.cuda()
    model.img_mean = np.array([0.485, 0.456, 0.406])
    model.img_std = np.array([0.229, 0.224, 0.225])

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

    eval_model(model,dataloader,exp_const)
        