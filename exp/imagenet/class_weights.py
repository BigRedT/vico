import os
import copy
import numpy as np
from tqdm import tqdm

import utils.io as io
from utils.constants import save_constants
from .dataset import ImagenetNoImgsDataset


def main(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    save_constants(
        {'exp': exp_const,'data': data_const},
        exp_const.exp_dir)

    print('Creating dataloader ...')
    data_const = copy.deepcopy(data_const)
    dataset = ImagenetNoImgsDataset(data_const)
    num_classes = len(dataset.wnids)
    num_samples = np.zeros([num_classes])
    num_pos = np.zeros([num_classes])
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        num_pos += data['label_vec']
        num_samples += data['weight_vec']

    num_neg = num_samples - num_pos
    pos_weight = num_samples / (2*num_pos + 1e-6)
    neg_weight = num_samples / (2*num_neg + 1e-6)
    class_weights = np.stack((pos_weight,neg_weight),1)

    class_weights_npy = os.path.join(exp_const.exp_dir,'class_weights.npy')
    np.save(class_weights_npy,class_weights)