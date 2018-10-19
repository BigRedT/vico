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
from exp.glove.concat_embed_dataset import ConcatEmbedDataset
from exp.glove.models.encoder import Encoder


def get_embeddings(model,dataloader,exp_const):
    model.encoder.eval()
    
    print('Initializing embeddings matrix ...')
    embeddings = np.zeros(
        [len(dataloader.dataset),model.encoder.const.output_dims],
        dtype=np.float32)

    print('Begin encoding ...')
    for it, data in enumerate(tqdm(dataloader)):
        concat_embeddings = Variable(data['embedding']).cuda()
        _,embedding = model.encoder(concat_embeddings)
        embedding = embedding.data.cpu()
        for i,word in enumerate(data['word']):
            idx = dataloader.dataset.word_to_idx[word]
            embeddings[idx] = embedding[i]

    return embeddings


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
    dataset = ConcatEmbedDataset(data_const)
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True)

    print('Get embeddings ...')
    embeddings = get_embeddings(model,dataloader,exp_const)

    print('Save embeddings h5py ...')
    visual_word_vecs_h5py = h5py.File(
        os.path.join(exp_const.exp_dir,'visual_word_vecs.h5py'),
        'w')
    visual_word_vecs_h5py.create_dataset(
        'embeddings',
        data=embeddings,
        chunks=(1,embeddings.shape[1]))
    visual_word_vecs_h5py.close()
    
    print('Save embeddings word idx json ...')
    visual_word_vecs_idx_json = os.path.join(
        exp_const.exp_dir,
        'visual_word_vecs_idx.json')
    io.dump_json_object(
        dataloader.dataset.word_to_idx,
        visual_word_vecs_idx_json)