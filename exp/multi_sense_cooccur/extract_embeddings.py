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
from .models.logbilinear import LogBilinear
from .dataset import MultiSenseCooccurDataset


def main(exp_const,data_const,model_const):
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.net = LogBilinear(model.const.net)
    if model.const.model_num is not None:
        model.net.load_state_dict(torch.load(model.const.net_path))
    
    embeddings = 0.5*(model.net.embed1.W.weight + model.net.embed2.W.weight)
    embeddings = embeddings.data.numpy()
    embeddings_json = os.path.join(exp_const.exp_dir,'visual_embeddings.npy')
    np.save(embeddings_json,embeddings)

    print('Saving word_to_idx.json ...')
    dataset = MultiSenseCooccurDataset(data_const)
    word_to_idx = dataset.word_to_idx
    word_to_idx_json = os.path.join(exp_const.exp_dir,'word_to_idx.json')
    io.dump_json_object(word_to_idx,word_to_idx_json)


