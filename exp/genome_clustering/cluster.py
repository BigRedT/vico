import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
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
    object_ids = io.load_json_object(data_const.object_ids_json)
    image_ids = io.load_json_object(data_const.image_ids_json)

    print('Creating dataloader ...')
    # dataloader = DataLoader(
    #     FeatsDataset(feats),
    #     batch_size=exp_const.batch_size,
    #     collate_fn=collate_fn,
    #     shuffle=True,
    #     drop_last=True)
    print('Num samples',feats.shape[0])

    print('Clustering ...')
    clustering = MiniBatchKMeans(
        n_clusters=exp_const.n_clusters,
        batch_size=exp_const.batch_size,
        max_iter=exp_const.num_epochs,
        verbose=10,
        n_init=1,
        max_no_improvement=None)
    cluster_ids = clustering.fit_predict(feats)
    
    print('Done clustering!')
    cluster_ids_json = os.path.join(exp_const.exp_dir,'cluster_ids.json')
    io.dump_json_object(cluster_ids,cluster_ids_json)

    cluster_id_to_feat_ids = {}
    for i,cid in enumerate(cluster_ids):
        cid = str(cid)
        if cid not in cluster_id_to_feat_ids:
            cluster_id_to_feat_ids[cid] = []
        
        cluster_id_to_feat_ids[cid].append(i)

    cluster_id_to_feat_ids_json = os.path.join(
        exp_const.exp_dir,
        'cluster_id_to_feat_ids.json')
    io.dump_json_object(cluster_id_to_feat_ids,cluster_id_to_feat_ids_json)


