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

    print('Computing transformed embeddings ...')
    xformed_embeddings = []
    for cooccur_type in exp_const.cooccur_types:
        xform = getattr(model.net,f'xform_{cooccur_type}')
        xformed_embeddings.append(xform(embeddings).cpu().data.numpy())
        print(cooccur_type,xformed_embeddings[-1].shape)
        
    xformed_embeddings = np.concatenate(xformed_embeddings,1)
    print('Concatenated xformed embedding shape',xformed_embeddings.shape)
    xformed_embeddings_json = os.path.join(
        exp_const.exp_dir,
        'visual_embeddings_xformed.npy')
    np.save(xformed_embeddings_json,xformed_embeddings)


