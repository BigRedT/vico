import os
import h5py
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
from exp.glove.visual_features_dataset import VisualFeaturesDataset
from exp.glove.models.encoder import Encoder


def get_visual_features(model,dataloader,exp_const):
    model.encoder.eval()
    
    print('Initializing visual features matrix ...')
    visual_features = np.zeros(
        [len(dataloader.dataset),model.encoder.const.output_dims],
        dtype=np.float32)

    print('Begin encoding ...')
    for it, data in enumerate(tqdm(dataloader)):
        features = Variable(data['feature']).cuda()
        _, ae_features = model.encoder(features)
        ae_features = ae_features.data.cpu()
        for i,word in enumerate(data['word']):
            idx = dataloader.dataset.word_to_idx[word]
            visual_features[idx] = ae_features[i]

    return visual_features


def main(exp_const,data_const,model_const):
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.encoder = Encoder(model.const.encoder).cuda()
    encoder_path = os.path.join(
        exp_const.model_dir,
        'encoder_' + str(model.const.model_num))
    model.encoder.load_state_dict(torch.load(encoder_path))

    print('Creating dataloader ...')
    dataset = VisualFeaturesDataset(data_const)
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True)

    print('Get features ...')
    features = get_visual_features(model,dataloader,exp_const)

    print('Save features h5py ...')
    word_features_h5py = h5py.File(
        os.path.join(exp_const.exp_dir,'word_features.h5py'),
        'w')
    word_features_h5py.create_dataset(
        'features',
        data=features,
        chunks=(1,features.shape[1]))
    word_features_h5py.create_dataset(
        'mean',
        data=np.mean(features,axis=0))
    word_features_h5py.close()
    
    print('Save features word idx json ...')
    word_to_idx_json = os.path.join(
        exp_const.exp_dir,
        'word_to_idx.json')
    io.dump_json_object(
        dataloader.dataset.word_to_idx,
        word_to_idx_json)