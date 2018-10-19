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


class ConcatEmbedDatasetConstants(io.JsonSerializableClass):
    def __init__(
            self,
            concat_dir):
        super(ConcatEmbedDatasetConstants,self).__init__()
        self.concat_dir = concat_dir
        self.embeddings_h5py = os.path.join(
            self.concat_dir,
            'visual_word_vecs.h5py')
        self.word_to_idx_json = os.path.join(
            self.concat_dir,
            'visual_word_vecs_idx.json')
        self.glove_dim = 300


class ConcatEmbedDataset(Dataset):
    def __init__(self,const):
        self.const = copy.deepcopy(const)
        self.embeddings = self.load_embeddings(self.const.embeddings_h5py)
        self.word_to_idx, self.idx_to_word = \
            self.load_word_idx(self.const.word_to_idx_json)

    def load_embeddings(self,embeddings_h5py):
        embeddings = h5py.File(embeddings_h5py,'r')['embeddings'][()]
        return embeddings

    def load_word_idx(self,word_to_idx_json):
        word_to_idx = io.load_json_object(word_to_idx_json)
        idx_to_word = {v:k for k,v in word_to_idx.items()}
        return word_to_idx, idx_to_word

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self,i):
        word = self.idx_to_word[i]
        embedding = self.embeddings[i].astype(np.float32)
        to_return = {
            'embedding': embedding,
            #'glove': embedding[:self.const.glove_dim],
            #'visual': embedding[self.const.glove_dim:],
            'word': word,
        }
        return to_return


if __name__=='__main__':
    concat_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/' + \
        'normalized_resnet_embeddings_recon_loss_trained_on_google/' + \
        'concat_glove_and_visual')
    data_const = ConcatEmbedDatasetConstants(concat_dir)
    dataset = ConcatEmbedDataset(data_const)
    dataloader = DataLoader(dataset,batch_size=100)
    for data in dataloader:
        import pdb; pdb.set_trace()

