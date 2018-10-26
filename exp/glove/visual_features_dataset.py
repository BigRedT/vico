import os
import copy
import glob
import h5py
import itertools
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import utils.io as io
from torch.utils.data.dataloader import default_collate


class VisualFeaturesDatasetConstants(io.JsonSerializableClass):
    def __init__(
            self,
            features_dir):
        super(VisualFeaturesDatasetConstants,self).__init__()
        self.features_dir = features_dir
        self.features_h5py = os.path.join(
            self.features_dir,
            'word_features.h5py')
        self.word_to_idx_json = os.path.join(
            self.features_dir,
            'word_to_idx.json')


class VisualFeaturesDataset(Dataset):
    def __init__(self,const):
        super(VisualFeaturesDataset,self).__init__()
        self.const = copy.deepcopy(const)
        self.features = self.load_features(self.const.features_h5py)
        self.word_to_idx, self.idx_to_word = \
            self.load_word_idx(self.const.word_to_idx_json)

    def load_features(self,features_h5py):
        features = h5py.File(features_h5py,'r')['features'][()]
        return features

    def load_word_idx(self,word_to_idx_json):
        word_to_idx = io.load_json_object(word_to_idx_json)
        idx_to_word = {v:k for k,v in word_to_idx.items()}
        return word_to_idx, idx_to_word

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self,i):
        word = self.idx_to_word[i]
        feature = self.features[i].astype(np.float32)
        to_return = {
            'feature': feature,
            'word': word,
        }
        return to_return


if __name__=='__main__':
    feature_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/' + \
        'normalized_resnet_features_recon_loss_trained_on_google')
    data_const = VisualFeaturesDatasetConstants(feature_dir)
    dataset = VisualFeaturesDataset(data_const)
    dataloader = DataLoader(dataset,batch_size=100,shuffle=True)
    for data in dataloader:
        import pdb; pdb.set_trace()

