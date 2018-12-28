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


class CooccurDataset(Dataset):
    def __init__(self,const):
        super(CooccurDataset,self).__init__()
        self.const = copy.deepcopy(const)
        self.cooccur = io.load_json_object(self.const.cooccur_json)
        self.words = sorted(list(self.cooccur.keys()))
        self.word_to_idx = {word:idx for idx,word in enumerate(self.words)}
        self.cooccur_list = self.create_cooccur_list()

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

    def __getitem__(self,i):
        idx1,idx2,x = self.cooccur_list[i]
        to_return = {
            'idx1': idx1,
            'idx2': idx2,
            'x': x,
            'word1': self.words[idx1],
            'word2': self.words[idx2],
        }
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
        batch_size=10,
        shuffle=True,
        collate_fn=collate_fn)
    for data in dataloader:
        import pdb; pdb.set_trace()