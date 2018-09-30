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


def topk_accuracy(scores,labels,k=10):
    _,pred_labels = torch.topk(scores,k,dim=1)
    pred_labels = pred_labels.data.cpu().numpy().tolist()
    gt_values,gt_labels = torch.topk(labels,k,dim=1)
    gt_labels = gt_labels.data.cpu().numpy().tolist()
    gt_values = gt_values.data.cpu().numpy().tolist()
    B = scores.size(0)
    accuracy = 0
    for i in range(B):
        gt_labels_ = set()
        for v,l in zip(gt_values[i],gt_labels[i]):
            if v == 1.0:
                gt_labels_.add(l)
            else:
                break
        if len(set(pred_labels[i]) & gt_labels_) > 0:
            accuracy += 1
    accuracy = accuracy / B
    return accuracy


def eval_model(exp_const,dataloader,model):
    model.net.eval()
    model.object_classifier_layers.eval()
    model.attribute_classifier_layers.eval()
    sigmoid = pytorch_layers.get_activation('Sigmoid')
    k = 0
    object_accuracy = 0
    attribute_accuracy = 0
    total_samples = 0
    for data in tqdm(dataloader):
        if k >= exp_const.num_eval_iters:
            break
        if data is None:
            continue
        
        regions = dataloader.dataset.normalize(
            data['regions']/255,
            model.img_mean,
            model.img_std)
        regions = Variable(torch.FloatTensor(regions).cuda(),volatile=True)
        regions = regions.permute(0,3,1,2)
        object_labels = Variable(torch.LongTensor(
            data['object_labels']).cuda(),volatile=True)
        object_labels_idx = Variable(torch.LongTensor(
            data['object_labels_idx']).cuda(),volatile=True)
        attribute_labels = Variable(torch.FloatTensor(
            data['attribute_labels']).cuda(),volatile=True)
        B = regions.size(0)
        rB = exp_const.region_batch_size
        for i in range(math.ceil(B/rB)):
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
            samples = object_scores.size(0)
            total_samples += samples
            object_accuracy += samples*topk_accuracy(
                object_scores,
                object_labels[i*rB:r],
                k=100)
            # attribute_probs = sigmoid(attribute_scores) > 0.00001
            # attribute_probs = attribute_probs.float()
            attribute_accuracy += samples*topk_accuracy(
                attribute_scores,
                attribute_labels[i*rB:r],
                k=100)
    object_accuracy = object_accuracy / total_samples
    attribute_accuracy = attribute_accuracy / total_samples
    results = {
        'top_100_accuracy': {
            'object': object_accuracy,
            'attribute': attribute_accuracy
        }
    }
    results_json = os.path.join(
        exp_const.exp_dir,
        f'results_{model.const.model_num}.json')
    io.dump_json_object(results,results_json)
    print(results)


def main(exp_const,data_const,model_const):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    print('Creating network ...')
    model = Model()
    model.const = model_const
    if exp_const.use_resnet_normalized == True:
        model.net = resnet152_normalized(pretrained=False)
    else:
        model.net = resnet152(pretrained=False)
    model.object_classifier_layers = WordClassifierLayer(
        model_const.object_classifier_layers)
    model.attribute_classifier_layers = WordClassifierLayer(
        model_const.attribute_classifier_layers)
    model.net.load_state_dict(torch.load(model.const.net_path))
    model.object_classifier_layers.load_state_dict(torch.load(
        model.const.object_classifier_layers_path))
    model.attribute_classifier_layers.load_state_dict(torch.load(
        model.const.attribute_classifier_layers_path))
    model.net.cuda()
    model.object_classifier_layers.cuda()
    model.attribute_classifier_layers.cuda()
    model.img_mean = np.array([0.485, 0.456, 0.406])
    model.img_std = np.array([0.229, 0.224, 0.225])
    

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

    print('Evaluating model ...')
    eval_model(exp_const,dataloader,model)