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
from exp.semeval_2018_10.models.concat_svm_simple import ConcatSVM
from exp.semeval_2018_10.dataset import SemEval201810Dataset
from exp.semeval_2018_10.f1_computer import compute_f1
from utils.pytorch_layers import set_learning_rate


def train_model(model,train_data_loader,val_data_loader,exp_const):
    model.concat_svm.mlp.layers[0][0].weight = nn.Parameter(
        0*model.concat_svm.mlp.layers[0][0].weight.data)
    params = model.concat_svm.parameters()
    optimizer = optim.SGD(params,lr=exp_const.lr,momentum=0.9)

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
        lr = exp_const.lr*0.5**(epoch//1)
        set_learning_rate(optimizer,lr=lr)
        for i, data in enumerate(train_data_loader):
            step += 1
            
            model.concat_svm.train()
            score = model.concat_svm(
                Variable(data['word1_embedding']).cuda(),
                Variable(data['word2_embedding']).cuda(),
                Variable(data['feature_embedding']).cuda())
            total_loss,hinge_loss,l2_reg = model.concat_svm.compute_loss(
                score,
                Variable(data['label']).cuda())

            train_scores_tuples, train_best_scores_tuple = \
                compute_f1(
                    score.data.cpu().numpy(),
                    data['label'].numpy(),
                    np.arange(-0.4,0.4,0.05)) #-1.2,1.2,0.2
            train_avg_f1, train_pos_f1, train_neg_f1, \
                train_acc, train_best_thresh = train_best_scores_tuple
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step%1==0:
                w = model.concat_svm.w.data
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    'Total Loss: {:.4f} | ' + \
                    'Hinge Loss: {:.4f} | ' + \
                    'L2 Loss: {:.4f} | ' + \
                    'Avg F1: {:.4f} | ' + \
                    'Acc: {:.4f} | ' + \
                    'LR: {:.4f} | ' + \
                    'w_glove: {:.4f} | ' + \
                    'w_visual: {:.4f} | ' + \
                    'b: {:.4f}'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    total_loss.data[0],
                    hinge_loss.data[0],
                    l2_reg.data[0],
                    train_avg_f1,
                    train_acc,
                    lr,
                    w[0],
                    w[1],
                    w[2])
                print(log_str)
                log_value('train_loss',total_loss.data[0],step)
                log_value('train_hinge_loss',hinge_loss.data[0],step)
                log_value('train_l2_reg',l2_reg.data[0],step)
                log_value('train_f1',train_avg_f1,step)
                log_value('train_acc',train_acc,step)

            if step%10==0:
                val_hinge_loss, val_scores_tuples, val_best_scores_tuple = \
                    eval_model(model,val_data_loader,exp_const)
                val_avg_f1, val_pos_f1, val_neg_f1, val_acc, val_best_thresh = \
                    val_best_scores_tuple
                
                # Log on tensorboard
                log_value('val_hinge_loss',val_hinge_loss,step)
                log_value('val_f1',val_avg_f1,step)
                log_value('val_acc',val_acc,step)
                log_value('val_best_thresh',val_best_thresh,step)
                
                # Print on screen
                log_str = \
                    'val f1: {:.4f} | ' + \
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
                print(exp_const.exp_name)

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
                    model.concat_svm.state_dict(),
                    model_pth)


def eval_model(model,data_loader,exp_const):
    model.concat_svm.eval()
    pred_score = []
    gt_label = []
    hinge_loss = 0
    count = 0
    for i, data in enumerate(tqdm(data_loader)):
        score = model.concat_svm(
            Variable(data['word1_embedding']).cuda(),
            Variable(data['word2_embedding']).cuda(),
            Variable(data['feature_embedding']).cuda())
        batch_hinge_loss = model.concat_svm.compute_hinge_loss(
            score,
            Variable(data['label']).cuda())   
        
        batch_size = score.shape[0]
        count += batch_size
        hinge_loss += (batch_hinge_loss.data[0]*batch_size)
        pred_score.append(score.data.cpu().numpy())
        gt_label.append(data['label'].numpy())
    
    hinge_loss = hinge_loss / count
    #loss = loss + model.concat_svm.compute_l2_loss().data[0]
    pred_score = np.concatenate(pred_score)
    gt_label = np.concatenate(gt_label)
    scores_tuples, best_scores_tuple = compute_f1(
        pred_score,
        gt_label,
        np.arange(-0.4,0.4,0.05)) #-1.2,1.2,0.2
    return hinge_loss, scores_tuples, best_scores_tuple
        

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
    model.concat_svm = ConcatSVM(model_const.concat_svm).cuda()
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
