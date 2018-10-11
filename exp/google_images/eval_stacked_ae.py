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
from PIL import Image

import utils.io as io
from utils.html_writer import HtmlWriter
from utils.model import Model
from utils.constants import save_constants
import utils.pytorch_layers as pytorch_layers
from exp.google_images.dataset_image_level import GoogleImagesImageLevelDataset
from exp.google_images.models.resnet_encoder_outer import resnet34
from exp.google_images.models.resnet_encoder_inner import ResnetEncoderInner
from exp.google_images.models.word_classification_mlp import WordClassifierLayer
from exp.google_images.models.decoder import Decoder


def topk_accuracy(scores,gt,k):
    _,idxs = torch.topk(scores,k,1)
    idxs = idxs.data.cpu().numpy()
    gt = gt.data.cpu().numpy()
    B = idxs.shape[0]
    count = 0
    for i in range(B):
        if gt[i] in set(idxs[i].tolist()):
            count += 1
    accuracy = count / B
    return accuracy


def eval_model(exp_const,dataloader,model):
    step = 0
    total = 0
    correct_top_1 = 0
    correct_top_100 = 0
    for it,data in enumerate(tqdm(dataloader)):
        step += 1
        
        regions_unnormalized = \
            Variable(torch.FloatTensor(data['img']).cuda(),volatile=True)
        regions = dataloader.dataset.normalize(
            regions_unnormalized/255,
            model.img_mean,
            model.img_std)
        regions = regions.permute(0,3,1,2)
        regions_unnormalized = regions_unnormalized.permute(0,3,1,2)
        word_idx = Variable(torch.LongTensor(data['idx']).cuda(),volatile=True)
        
        resnet_feats = model.encoder_outer(regions)
        x, x_norm = model.encoder_inner(resnet_feats)
        if exp_const.use_resnet_normalized == True:
            embedding = x_norm
        else:
            embedding = x
            
        word_scores = model.word_classifier_layers(
            embedding)
        
        B = word_scores.size(0)
        total += B
        correct_top_1 += B*topk_accuracy(word_scores,word_idx,1)
        correct_top_100 += B*topk_accuracy(word_scores,word_idx,100)

    accuracy_top_1 = correct_top_1 / total
    accuracy_top_100 = correct_top_100 / total
    to_return = {
        'top_1_accuracy': accuracy_top_1,
        'top_100_accuracy': accuracy_top_100,
    }
    return to_return


def main(exp_const,data_const,model_const):
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.encoder_outer = resnet34(pretrained=True)
    model.encoder_inner = ResnetEncoderInner(model.const.encoder_inner)
    model.word_classifier_layers = WordClassifierLayer(
        model_const.word_classifier_layers)
    model.encoder_outer.load_state_dict(
        torch.load(model.const.encoder_outer_path))
    model.encoder_inner.load_state_dict(
        torch.load(model.const.encoder_inner_path))
    model.word_classifier_layers.load_state_dict(torch.load(
            model.const.word_classifier_layers_path))
    model.encoder_outer.cuda()
    model.encoder_inner.cuda()
    model.encoder_outer.eval()
    model.encoder_inner.eval()
    model.word_classifier_layers.cuda()
    model.word_classifier_layers.eval()
    model.img_mean = Variable(
        torch.FloatTensor(np.array([0.485, 0.456, 0.406]))).cuda()
    model.img_std = Variable(
        torch.FloatTensor(np.array([0.229, 0.224, 0.225]))).cuda()

    print('Creating dataloader ...')
    dataset = GoogleImagesImageLevelDataset(data_const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    results = eval_model(exp_const,dataloader,model)
    print(results)
    results_json = os.path.join(
        exp_const.exp_dir,
        f'results_{model.const.model_num}.json')
    io.dump_json_object(results,results_json)