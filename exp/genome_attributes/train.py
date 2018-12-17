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
from .models.resnet_normalized import ResnetNormalizedModel
from .dataset import GenomeAttributesDataset
from .vis.vis_sim_mat import create_sim_heatmap


class L2RegLoss(nn.Module):
    def __init__(self):
        super(L2RegLoss,self).__init__()
        
    def forward(self,x):
        return torch.mean(torch.sum(x*x,1))


class Norm1Loss(nn.Module):
    def __init__(self):
        super(Norm1Loss,self).__init__()
        
    def forward(self,x):
        norm_sq = torch.sum(x*x,1)
        loss = torch.mean((norm_sq-1.0)*(norm_sq-1.0))
        return loss


class PosNllLoss(nn.Module):
    def __init__(self):
        super(PosNllLoss,self).__init__()

    def forward(self,x,y):
        return -torch.sum(y*torch.log(x+1e-6))/(torch.sum(y)+1e-6)


class MarginLoss(nn.Module):
    def __init__(self,margin=0.5):
        self.margin = margin
        super(MarginLoss,self).__init__()

    def forward(self,x,y,pos_ids):
        B = x.size(0)
        D = x.size(1)
        loss = 0
        count = 0
        num_pos = 0
        for i in range(B):
            num_pos += len(pos_ids[i])
            for k in pos_ids[i]:
                loss += torch.sum((1-y[i])*torch.max(0*x[i],self.margin+x[i]-x[i,k]))
                count += (D - len(pos_ids[i]))
        loss = loss / count
        return loss


def vecidx2matidx(idx,C):
    rows = idx//C
    cols = idx - C*rows
    return rows,cols


def compute_classifier_sim_mat(model,k=20):
    # classifier_weights: num_classes x feat_dim
    classifier_weights = model.net.resnet_layers.fc.weight.data
    weight_norm = torch.norm(
        classifier_weights,
        p=2,
        dim=1,
        keepdim=True)
    classifier_weights = classifier_weights / (weight_norm+1e-6)
    sim_mat = torch.matmul(classifier_weights,classifier_weights.permute(1,0))
    
    C = sim_mat.size(0)

    sim_mat[range(C),range(C)] = -1
    _,idx1 = torch.topk(sim_mat.view(-1),k,largest=True)
    rows1,cols1 = vecidx2matidx(idx1.cpu().numpy(),C)
    rows1 = set(rows1)
    cols1 = set(cols1)

    sim_mat[range(C),range(C)] = 1
    _,idx2 = torch.topk(sim_mat.view(-1),k,largest=False)
    rows2,cols2 = vecidx2matidx(idx2.cpu().numpy(),C)
    rows2 = set(rows2)
    cols2 = set(cols2)

    rows = sorted(list(rows1.union(rows2)))
    cols = sorted(list(cols1.union(cols2)))
    sim_mat_sm = sim_mat[np.array(rows)[:,None],np.array(cols)]
    return sim_mat_sm,rows,cols


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

    pos_nll_criterion = PosNllLoss()
    margin_criterion = MarginLoss(margin=exp_const.margin)
    l2_reg_criterion = Norm1Loss()
    #l2_reg_criterion = L2RegLoss()
    bce_criterion = nn.BCELoss()

    sigmoid = nn.Sigmoid()

    step = 0
    if model.const.model_num is not None:
        step = model.const.model_num

    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean))
    img_std = Variable(torch.cuda.FloatTensor(model.img_std))

    sim_mat_vis_dir = os.path.join(exp_const.vis_dir,'train_sim_mat')
    io.mkdir_if_not_exists(sim_mat_vis_dir,recursive=True)

    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloader):
            # Set mode
            model.net.train()

            # Forward pass
            imgs = Variable(data['regions'].cuda().float()/255.)
            imgs = dataloader.dataset.normalize(
                imgs,
                img_mean,
                img_std)
            imgs = imgs.permute(0,3,1,2)
            logits, last_layer_feats_normalized, last_layer_feats = \
                model.net(imgs)
            prob = sigmoid(logits)
            
            # Computer loss
            attr_labels = Variable(data['attribute_labels'].cuda())
            attr_labels_idxs = data['attribute_labels_idxs']
            # pos_nll_loss = 0*pos_nll_criterion(prob,attr_labels)
            # margin_loss = 0*margin_criterion(prob,attr_labels,attr_labels_idxs)
            l2_reg_loss = l2_reg_criterion(last_layer_feats)
            bce_loss = bce_criterion(prob,attr_labels)
            loss = bce_loss + 1e-6*l2_reg_loss #+ pos_nll_loss + margin_loss
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                log_items = {
                    'Loss': loss.data[0],
                    # 'Pos NLL Loss': pos_nll_loss.data[0],
                    # 'Margin Loss': margin_loss.data[0],
                    'BCE Loss': bce_loss.data[0],
                    'L2 Reg': l2_reg_loss.data[0],
                    'LR': lr,
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
                sim_mat_sm,rows,cols = compute_classifier_sim_mat(model)
                attr_synsets = dataloader.dataset.sorted_attribute_synsets
                xlabels = [attr_synsets[i] for i in cols]
                ylabels = [attr_synsets[i] for i in rows]
                sim_mat_html = os.path.join(
                    sim_mat_vis_dir,
                    f'{step}.html')
                create_sim_heatmap(
                    sim_mat_sm.cpu().numpy(),
                    xlabels,
                    ylabels,
                    sim_mat_html)

            step += 1

        lr = 0.5*lr
        pytorch_layers.set_learning_rate(opt,lr)


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

    train_model(model,dataloader,exp_const)