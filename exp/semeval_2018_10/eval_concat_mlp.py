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
from exp.semeval_2018_10.model_selection import select_best_concat_mlp


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
    _, best_scores_tuple = compute_f1(
        pred_prob,
        gt_label,
        np.array([model.concat_mlp.const.thresh]))
    result = {
        'loss': loss,
        'avg_f1': best_scores_tuple[0],
        'pos_f1': best_scores_tuple[1],
        'neg_f1': best_scores_tuple[2],
        'acc': best_scores_tuple[3],
        'thresh': best_scores_tuple[4]
    }
    return result
        

def main(exp_const,data_const,model_const):
    print('Creating model ...')
    model = Model()
    model.const = model_const
    model.concat_mlp = ConcatMLP(model_const.concat_mlp).cuda()

    print('Select best model ...')
    step_to_val_best_scores_tuple_json = os.path.join(
        exp_const.exp_dir,
        'step_to_val_best_scores_tuple.json')
    step_to_val_best_scores_tuple = io.load_json_object(
        step_to_val_best_scores_tuple_json)
    best_step, best_scores_tuple = select_best_concat_mlp(
        step_to_val_best_scores_tuple)
    model.concat_mlp.const.thresh = best_scores_tuple[3]
    model_pth = os.path.join(
        exp_const.model_dir,f'{best_step}')
    model.concat_mlp.load_state_dict(torch.load(model_pth))
    print(f'Selcted model at step: {best_step}')
    print(f'Selected thresh: {model.concat_mlp.const.thresh}')

    print('Creating data loader ...')
    data_const = copy.deepcopy(data_const)
    data_loader = DataLoader(
        SemEval201810Dataset(data_const),
        batch_size=exp_const.batch_size,
        shuffle=False)

    print('Begin evaluation ...')
    result = eval_model(model,data_loader,exp_const)
    result_json = os.path.join(exp_const.exp_dir,'results.json')
    io.dump_json_object(result,result_json)
    print(io.dumps_json_object(result))
