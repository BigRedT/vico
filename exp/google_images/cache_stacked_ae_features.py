import os
import glob
import h5py
import math
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from PIL import Image

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
from exp.google_images.models.resnet_encoder_outer import resnet34
from exp.google_images.models.resnet_encoder_inner import ResnetEncoderInner
from exp.google_images.dataset import GoogleImagesDataset



def main(exp_const,model_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    save_constants({
        'exp': exp_const,
        'data': data_const,
        'model': model_const
        },exp_const.exp_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.encoder_outer = resnet34(pretrained=True)
    model.encoder_inner = ResnetEncoderInner(model.const.encoder_inner)
    model.encoder_outer.load_state_dict(torch.load(model.const.encoder_outer_path))
    model.encoder_inner.load_state_dict(torch.load(model.const.encoder_inner_path))
    model.encoder_outer.cuda()
    model.encoder_inner.cuda()
    model.encoder_outer.eval()
    model.encoder_inner.eval()

    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])

    print('Creating word_features.h5py ...')
    word_features_h5py = h5py.File(
        os.path.join(exp_const.exp_dir,'word_features.h5py'),
        'w')
    vocab = io.load_json_object(data_const.vocab_json)
    num_words = len(vocab)
    print(f'Num words: {num_words}')

    print('Creating dataset ...')
    dataset = GoogleImagesDataset(data_const)
    word_to_idx = {}
    features = np.zeros([num_words,exp_const.feature_dim])
    for i,word in enumerate(tqdm(vocab.keys())):
        data = dataset[word]
        imgs = dataset.normalize(
            data['imgs']/255,
            img_mean,
            img_std)
        imgs = np.transpose(imgs,[0,3,1,2])
        imgs = Variable(torch.FloatTensor(imgs).cuda(),volatile=True)
        x,x_norm = model.encoder_inner(model.encoder_outer(imgs))
        features[i] = torch.mean(x_norm,0).data.cpu().numpy()
        word_to_idx[word] = i

    word_features_h5py.create_dataset(
        'features',
        data=features,
        chunks=(1,exp_const.feature_dim))
    word_features_h5py.create_dataset(
        'mean',
        data=np.mean(features,0))
    word_features_h5py.close()

    word_to_idx_json = os.path.join(exp_const.exp_dir,'word_to_idx.json')
    io.dump_json_object(word_to_idx,word_to_idx_json)