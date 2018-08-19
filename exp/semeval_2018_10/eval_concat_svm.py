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
from exp.semeval_2018_10.models.concat_svm import ConcatSVM
from exp.semeval_2018_10.dataset import SemEval201810Dataset
from exp.semeval_2018_10.f1_computer import compute_f1
from exp.semeval_2018_10.model_selection import select_best_concat_svm


def eval_model(model,data_loader,exp_const):
    model.concat_svm.eval()
    pred_score = []
    gt_label = []
    correct_preds = []
    incorrect_preds = []
    loss = 0
    count = 0
    for i, data in enumerate(tqdm(data_loader)):
        model.concat_svm.train()
        score = model.concat_svm(
            Variable(data['word1_embedding']).cuda(),
            Variable(data['word2_embedding']).cuda(),
            Variable(data['feature_embedding']).cuda())
        batch_loss = model.concat_svm.compute_hinge_loss(
            score,
            Variable(data['label']).cuda())   
        batch_size = score.shape[0]
        count += batch_size
        loss += (batch_loss.data[0]*batch_size)
        score = score.data.cpu().numpy()
        label = data['label'].numpy()
        pred_score.append(score)
        gt_label.append(label)
        for j in range(batch_size):
            sample = {
                'word1': data['word1'][j],
                'word2': data['word2'][j],
                'feature': data['feature'][j],
                'gt_label': float(data['label'][j] > 0.5),
                'pred_label': float(score[j] > model.concat_svm.const.thresh),
                'pred_score': score[j]
            }
            if (label[j] > 0.5) == (score[j] > model.concat_svm.const.thresh):
                 correct_preds.append(sample)
            else:
                incorrect_preds.append(sample)
    
    loss = loss / count
    pred_score = np.concatenate(pred_score)
    gt_label = np.concatenate(gt_label)
    _, best_scores_tuple = compute_f1(
        pred_score,
        gt_label,
        np.array([model.concat_svm.const.thresh]))
    result = {
        'loss': loss,
        'avg_f1': best_scores_tuple[0],
        'pos_f1': best_scores_tuple[1],
        'neg_f1': best_scores_tuple[2],
        'acc': best_scores_tuple[3],
        'thresh': best_scores_tuple[4]
    }
    return result, correct_preds, incorrect_preds
        

def main(exp_const,data_const,model_const):
    print('Creating model ...')
    model = Model()
    model.const = model_const
    model.concat_svm = ConcatSVM(model_const.concat_svm).cuda()

    print('Select best model ...')
    step_to_val_best_scores_tuple_json = os.path.join(
        exp_const.exp_dir,
        'step_to_val_best_scores_tuple.json')
    step_to_val_best_scores_tuple = io.load_json_object(
        step_to_val_best_scores_tuple_json)
    best_step, best_scores_tuple = select_best_concat_svm(
        step_to_val_best_scores_tuple)
    model.concat_svm.const.thresh = best_scores_tuple[4]
    model_pth = os.path.join(
        exp_const.model_dir,f'{best_step}')
    model.concat_svm.load_state_dict(torch.load(model_pth))
    print(f'Selcted model at step: {best_step}')
    print(f'Selected thresh: {model.concat_svm.const.thresh}')

    print('Creating data loader ...')
    data_const = copy.deepcopy(data_const)
    data_loader = DataLoader(
        SemEval201810Dataset(data_const),
        batch_size=exp_const.batch_size,
        shuffle=False)

    print('Begin evaluation ...')
    result, correct_preds, incorrect_preds = eval_model(model,data_loader,exp_const)
    result_json = os.path.join(
        exp_const.exp_dir,
        f'results_{data_const.subset}.json')
    io.dump_json_object(result,result_json)
    print(io.dumps_json_object(result))

    correct_preds_json = os.path.join(
        exp_const.exp_dir,
        f'correct_preds_{data_const.subset}.json')
    io.dump_json_object(correct_preds, correct_preds_json)

    incorrect_preds_json = os.path.join(
        exp_const.exp_dir,
        f'incorrect_preds_{data_const.subset}.json')
    io.dump_json_object(incorrect_preds, incorrect_preds_json)