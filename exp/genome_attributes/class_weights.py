import os
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils.io as io
from utils.constants import save_constants
from .dataset import GenomeAttributesNoImgsDataset


def main(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    save_constants(
        {'exp': exp_const,'data': data_const},
        exp_const.exp_dir)

    print('Creating dataloader ...')
    data_const = copy.deepcopy(data_const)
    dataset = GenomeAttributesNoImgsDataset(data_const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets','attribute_labels_idxs'])
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)
    num_classes = len(dataset.sorted_attribute_synsets)
    num_samples = np.zeros([num_classes])
    num_pos = np.zeros([num_classes])
    for data in tqdm(dataloader):
        if data is None:
            continue

        # if len(data['attribute_labels'])==0:
        #     continue
        labels = data['attribute_labels'].numpy()
        #labels = np.stack(data['attribute_labels'],0)
        num_pos += np.sum(labels,0,keepdims=False)
        num_samples += np.sum(0*labels+1,0,keepdims=False)

    num_neg = num_samples - num_pos
    pos_weight = num_samples / (2*num_pos + 1e-6)
    neg_weight = num_samples / (2*num_neg + 1e-6)
    class_weights = np.stack((pos_weight,neg_weight),1)

    class_weights_npy = os.path.join(exp_const.exp_dir,'class_weights.npy')
    np.save(class_weights_npy,class_weights)