import os
import copy
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

import utils.io as io


class CooccurDatasetConstants(io.JsonSerializableClass):
    def __init__(self):
        super(CooccurDatasetConstants,self).__init__()
        self.cooccur_json = None
        self.use_self_count = True
        self.alpha = 2


class CooccurDataset(Dataset):
    def __init__(self,const):
        super(CooccurDataset,self).__init__()
        self.const = copy.deepcopy(const)
        self.cooccur = io.load_json_object(self.const.cooccur_json)
        self.words = sorted(list(self.cooccur.keys()))
        self.word_to_idx = {word:idx for idx,word in enumerate(self.words)}
        self.cooccur_list = self.create_cooccur_list()
        self.pair_norm = self.get_pair_normalization()
        self.word_count = self.get_word_counts()

    def create_cooccur_list(self):
        print('Creating cooccurence list ...')
        cooccur_list = []
        for word1, context in tqdm(self.cooccur.items()):
            idx1 = self.word_to_idx[word1]
            for word2, count in context.items():
                if self.const.use_self_count==False and word1==word2:
                    continue
                idx2 = self.word_to_idx[word2]
                cooccur_list.append([idx1,idx2,count])

        return cooccur_list

    def __len__(self):
        return len(self.cooccur_list)

    def get_pair_normalization(self):
        print('Computing pair normalization ...')
        pair_norm = 0
        for word1, context in tqdm(self.cooccur.items()):
            for word2, count in context.items():
                if word1==word2:
                    continue
                pair_norm += count
        
        return pair_norm
    
    def get_word_counts(self):
        print('Computing word counts ...')
        word_count = {}
        for word1, context in tqdm(self.cooccur.items()):
            word_count[word1] = 0
            for word2, count in context.items():
                if word2==word1:
                    continue
                word_count[word1] += count

        return word_count

    def word_prob(self,word):
        eps = 1e-20
        num_words = len(self.cooccur)
        numer = self.word_count[word] + self.const.alpha*(num_words-0*1)
        denom = self.pair_norm + \
            self.const.alpha*num_words*(num_words-0*1) + \
            eps
        prob = numer / denom
        return prob

    def joint_prob(self,word1,word2):
        eps = 1e-20
        num_words = len(self.cooccur)
        numer = self.cooccur[word1][word2] + self.const.alpha
        denom = self.pair_norm + \
            self.const.alpha*num_words*(num_words-0*1) + \
            eps
        prob = numer / denom
        return prob

    def get_pmi(self,word1,word2):
        eps = 1e-20
        p_word1 = self.word_prob(word1)
        p_word2 = self.word_prob(word2)
        p_word1_word2 = self.joint_prob(word1,word2)
        exp_pmi = p_word1_word2 / (p_word1*p_word2+eps)
        pmi = np.log(exp_pmi + eps)
        return pmi

    def get_pmi2(self,word1,word2):
        eps = 1e-20
        p_word2 = self.word_count[word2] / (self.pair_norm + eps)
        p_word2_given_word1 = self.cooccur[word1][word2] / (self.word_count[word1] + eps)
        exp_pmi = p_word2_given_word1 / (p_word2+eps)
        pmi = np.log(exp_pmi + eps)
        return pmi

    def __getitem__(self,i):
        idx1,idx2,x = self.cooccur_list[i]
        to_return = {
            'idx1': idx1,
            'idx2': idx2,
            'x': x,
            'word1': self.words[idx1],
            'word2': self.words[idx2],
        }
        to_return['pmi'] = self.get_pmi(to_return['word1'],to_return['word2'])
        #to_return['pmi2'] = self.get_pmi2(to_return['word1'],to_return['word2'])
        return to_return

    def create_collate_fn(self):
        def collate_fn(batch):
            batch = [sample for sample in batch if sample is not None]
            if len(batch) == 0:
                return None
        
            batch_ = {}
            for k in batch[0].keys():
                batch_[k] = [sample[k] for sample in batch]

            return batch_

        return collate_fn


if __name__=='__main__':
    const = CooccurDatasetConstants()
    const.cooccur_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt/fused_word_cooccur.json')

    dataset = CooccurDataset(const)    
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=True,
        collate_fn=collate_fn)
    for data in dataloader:
        [print(p) for p in zip(
            data['word1'],data['word2'],data['x'],data['pmi'])]
        import pdb; pdb.set_trace()