import os
import h5py
import copy
import itertools
import numpy as np
from torch.utils.data import Dataset, DataLoader

import utils.io as io
from data.semeval_2018_10.constants import SemEval201810Constants


class SemEval201810DatasetConstants(SemEval201810Constants):
    def __init__(
            self,
            raw_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/semeval_2018_10/raw'),
            proc_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/semeval_2018_10/proc')):
        super(SemEval201810DatasetConstants,self).__init__(raw_dir,proc_dir)
        self.subset = ''
        self.embeddings_h5py = None
        self.word_to_idx_json = None
        self.random = False

    
class SemEval201810Dataset(Dataset):
    def __init__(self,const):
        super(SemEval201810Dataset,self).__init__()
        self.const = copy.deepcopy(const)
        self.embeddings = h5py.File(self.const.embeddings_h5py,'r')
        if self.const.random==True:
            n,d = self.embeddings['embeddings'].shape
            self.embeddings = {
                'embeddings': 2*(np.random.rand(n,d)-0.5)
            }
        self.word_to_idx = io.load_json_object(self.const.word_to_idx_json)
        self.samples = self.read_samples(self.const.subset)
        
    def read_samples(self,subset):
        if subset=='train':
            filename = self.const.train_json
        elif subset=='val':
            filename = self.const.val_json
        elif subset=='test':
            filename = self.const.truth_json
        else:
            msg = f'subset {subset} not supported'
            assert(False), msg
        return io.load_json_object(filename)

    def __len__(self):
        return len(self.samples)

    def get_embedding(self,word):
        if word not in self.word_to_idx:
            embedding_dim = self.embeddings['embeddings'].shape[1]
            embedding = np.zeros([embedding_dim],dtype=np.float32) 
        else:
            idx = self.word_to_idx[word]
            embedding = self.embeddings['embeddings'][idx].astype(np.float32)
        return embedding

    def __getitem__(self,i):
        sample = self.samples[i]
        word1,word2,feature,label = sample
        to_return = {
            'word1': word1,
            'word2': word2,
            'feature': feature,
            'label': np.float32(label),
            'word1_embedding': self.get_embedding(word1),
            'word2_embedding': self.get_embedding(word2),
            'feature_embedding': self.get_embedding(feature),
            #'mean_embedding': self.embeddings['mean'].value.astype(np.float32),
            #'std_embedding': self.embeddings['std'].value.astype(np.float32),
        }
        return to_return


if __name__=='__main__':
    from data.glove.constants import GloveConstantsFactory
    glove_const = GloveConstantsFactory.create()
    semeval_const = SemEval201810DatasetConstants()
    semeval_const.embeddings_h5py = glove_const.embeddings_h5py
    semeval_const.word_to_idx_json = glove_const.word_to_idx_json
    dataloader = DataLoader(
        SemEval201810Dataset(semeval_const),
        batch_size=4)
    for data in dataloader:
        import pdb; pdb.set_trace()

