import os
import copy
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
from exp.semeval_2018_10.models.concat_mlp import ConcatMLP
from exp.semeval_2018_10.dataset import SemEval201810Dataset
from exp.semeval_2018_10.f1_computer import compute_f1


def train_model(model,train_data_loader,val_data_loader,exp_const):
    params = model.concat_mlp.parameters()
    optimizer = optim.Adam(params,lr=exp_const.lr)

    criterion = nn.BCELoss()

    step_to_val_scores_tuples = {}
    step_to_val_best_scores_tuple = {}
    step_to_val_scores_tuples_json = os.path.join(
        exp_const.exp_dir,
        'step_to_val_scores_tuples.json')
    step_to_val_best_scores_tuple_json = os.path.join(
        exp_const.exp_dir,
        'step_to_val_best_scores_tuple.json')

    step = -1
    for epoch in range(exp_const.num_epochs):
        for i, data in enumerate(train_data_loader):
            step += 1
            
            model.concat_mlp.train()
            prob = model.concat_mlp(
                Variable(data['word1_embedding']).cuda(),
                Variable(data['word2_embedding']).cuda(),
                Variable(data['feature_embedding']).cuda())
            loss = criterion(
                prob,
                Variable(data['label']).cuda())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%10==0:
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    'Loss: {:.4f} | '
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    loss.data[0])
                print(log_str)
                log_value('train_loss',loss.data[0],step)

            if step%100==0:
                val_loss, val_scores_tuples, val_best_scores_tuple = \
                    eval_model(model,val_data_loader,exp_const)
                val_avg_f1, val_pos_f1, val_neg_f1, val_acc, val_best_thresh = \
                    val_best_scores_tuple
                
                # Log on tensorboard
                log_value('val_loss',val_loss,step)
                log_value('val_f1',val_avg_f1,step)
                log_value('val_acc',val_acc,step)
                log_value('val_best_thresh',val_best_thresh,step)
                
                # Print on screen
                log_str = \
                    'val avg f1: {:.4f} | ' + \
                    'pos_f1: {:.4f} | ' + \
                    'neg_f1: {:.4f} | ' + \
                    'acc: {:.4f} | ' + \
                    'thresh: {:.4f} |'
                log_str = log_str.format(
                    val_avg_f1,
                    val_pos_f1,
                    val_neg_f1,
                    val_acc,
                    val_best_thresh)
                print(log_str)

                # Save as json files to be used later for model selection
                step_to_val_scores_tuples[str(step)] = val_scores_tuples
                step_to_val_best_scores_tuple[str(step)] = val_best_scores_tuple
                io.dump_json_object(
                    step_to_val_scores_tuples,
                    step_to_val_scores_tuples_json)
                io.dump_json_object(
                    step_to_val_best_scores_tuple,
                    step_to_val_best_scores_tuple_json)

                # Save model
                model_pth = os.path.join(
                    exp_const.model_dir,
                    f'{step}')
                torch.save(
                    model.concat_mlp.state_dict(),
                    model_pth)


def eval_model(model,data_loader,exp_const):
    model.concat_mlp.eval()
    criterion = nn.BCELoss()
    pred_prob = []
    gt_label = []
    loss = 0
    count = 0
    for i, data in enumerate(tqdm(data_loader)):
        model.concat_mlp.train()
        prob = model.concat_mlp(
            Variable(data['word1_embedding']).cuda(),
            Variable(data['word2_embedding']).cuda(),
            Variable(data['feature_embedding']).cuda())
        batch_loss = criterion(
            prob,
            Variable(data['label']).cuda())   
        batch_size = prob.shape[0]
        count += batch_size
        loss += (batch_loss.data[0]*batch_size)
        pred_prob.append(prob.data.cpu().numpy())
        gt_label.append(data['label'].numpy())
    
    loss = loss / count
    pred_prob = np.concatenate(pred_prob)
    gt_label = np.concatenate(gt_label)
    scores_tuples, best_scores_tuple = compute_f1(pred_prob,gt_label)
    return loss, scores_tuples, best_scores_tuple
        

def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    configure(exp_const.log_dir)
    save_constants(
        {'exp':exp_const,'data':data_const,'model':model_const},
        exp_const.exp_dir)

    print('Creating model ...')
    model = Model()
    model.const = model_const
    model.concat_mlp = ConcatMLP(model_const.concat_mlp).cuda()
    model.to_txt(exp_const.exp_dir,single_file=True)

    print('Creating train data loader ...')
    train_data_const = copy.deepcopy(data_const)
    train_data_const.subset = 'train'
    train_data_loader = DataLoader(
        SemEval201810Dataset(train_data_const),
        batch_size=exp_const.batch_size,
        shuffle=True)

    print('Creating val data loader ...')
    val_data_const = copy.deepcopy(data_const)
    val_data_const.subset = 'val'
    val_data_loader = DataLoader(
        SemEval201810Dataset(val_data_const),
        batch_size=exp_const.batch_size,
        shuffle=False)

    print('Begin training ...')
    train_model(model,train_data_loader,val_data_loader,exp_const)
