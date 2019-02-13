import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
import fastcluster
from torch.utils.data import Dataset, DataLoader

import utils.io as io


class FeatsDataset(Dataset):
    def __init__(self,feats):
        self.feats = feats

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self,i):
        return self.feats[i]


def collate_fn(batch):
    return np.stack(batch)


def main(exp_const,data_const):
    print('Loading feats ...')
    feats = np.load(data_const.feats_npy)

    print('Creating dataloader ...')
    # dataloader = DataLoader(
    #     FeatsDataset(feats),
    #     batch_size=exp_const.batch_size,
    #     collate_fn=collate_fn,
    #     shuffle=True,
    #     drop_last=True)
    print('Num samples',feats.shape[0])

    print('Fitting knn ...')
    # nbrs = NearestNeighbors(
    #     n_neighbors=exp_const.k,
    #     algorithm='kd_tree',
    #     leaf_size=100,
    #     n_jobs=5)
    # nbrs.fit(feats)
    kdt = BallTree(
        feats,
        leaf_size=40,
        metric='euclidean')
    print('Fitted tree ...')
    indices = kdt.query(feats[:30],k=5,return_distance=False)
    #_,indices = nbrs.kneighbors(feats[:1000])
    print('Fitted knn!')

    indices_npy = os.path.join(exp_const.exp_dir,'knn_ids.npy')
    np.save(indices_npy,indices)


