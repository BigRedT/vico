import os
import pandas as pd
import copy
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

import utils.io as io


class MultiSenseCooccurDatasetConstants(io.JsonSerializableClass):
    def __init__(self):
        super(MultiSenseCooccurDatasetConstants,self).__init__()
        self.cooccur_csv = None
        self.use_self_count = False
        

class MultiSenseCooccurDataset(Dataset):
    def __init__(self,const):
        super(MultiSenseCooccurDataset,self).__init__()
        self.const = copy.deepcopy(const)
        print('Reading csv in pandas dataframe ...')
        self.df = pd.read_csv(self.const.cooccur_csv)
        self.column_names = self.df.columns.values.tolist()
        self.cooccur_types = self.column_names[2:]
        self.non_zero_cooccur = self.get_non_zero_cooccur()
        self.median_counts = self.get_median_counts()
        self.words = sorted([
            str(s) for s in list(set(self.df.word1.values.tolist()))])
        self.word_to_idx = {word:idx for idx,word in enumerate(self.words)}

    def get_non_zero_cooccur(self):
        non_zero_cooccur = {}
        for cooccur_type in self.cooccur_types:
            select_df = self.df[self.df[cooccur_type]>0]
            non_zero_cooccur[cooccur_type] = pd.DataFrame(data={
                'word1': select_df.word1.values,
                'word2': select_df.word2.values,
                'x': select_df[cooccur_type].values
            })
            min_cooccur = np.min(non_zero_cooccur[cooccur_type]['x'])
            max_cooccur = np.max(non_zero_cooccur[cooccur_type]['x'])
            num_cooccur = len(non_zero_cooccur[cooccur_type])
            print_str = f'{cooccur_type} | ' + \
                f'Min Non-Zero Cooccur: {min_cooccur} | ' + \
                f'Max Non-Zero Cooccur: {max_cooccur} | ' + \
                f'Num Non-Zero Cooccur: {num_cooccur}'
            print(print_str)
        
        return non_zero_cooccur

    def get_median_counts(self):
        median_counts = {}
        for cooccur_type in self.cooccur_types:
            x = self.df[cooccur_type]
            median_counts[cooccur_type] = np.median(x[x>0])

        return median_counts

    def __len__(self):
        len_ = 0
        for cooccur_type in self.cooccur_types:
            type_len = len(self.non_zero_cooccur[cooccur_type])
            if type_len > len_:
                len_ = type_len

        return len_

    def __getitem__(self,i):
        to_return = {}
        for cooccur_type in self.cooccur_types:
            i_ = i % len(self.non_zero_cooccur[cooccur_type])
            row = self.non_zero_cooccur[cooccur_type].iloc[i_]
            word1 = str(row['word1'])
            word2 = str(row['word2'])
            if word1==word2 and self.const.use_self_count==False:
                to_return_ = None
            else:
                to_return_ = {
                    'word1': word1,
                    'word2': word2, 
                    'idx1': self.word_to_idx[word1],
                    'idx2': self.word_to_idx[word2],
                    'x': float(row['x']),
                }
            to_return[cooccur_type] = to_return_
        
        return to_return

    def create_collate_fn(self):
        def collate_fn(batch_all_types):
            batch_all_types_ = {}
            for cooccur_type in self.cooccur_types:
                batch_all_types_[cooccur_type] = {
                    'word1': [],
                    'word2': [],
                    'idx1': [],
                    'idx2': [],
                    'x': []
                }
            
            for sample in batch_all_types:
                for cooccur_type in self.cooccur_types:
                    sample_ = sample[cooccur_type]
                    if sample_ is None:
                        continue

                    type_batch = batch_all_types_[cooccur_type]
                    for k,v in sample_.items():
                        type_batch[k].append(v)

            return batch_all_types_

        return collate_fn


if __name__=='__main__':
    const = MultiSenseCooccurDatasetConstants()
    const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt/merged_cooccur_self.csv')

    dataset = MultiSenseCooccurDataset(const)    
    import pdb; pdb.set_trace()
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        collate_fn=collate_fn)
    for data in dataloader:
        import pdb; pdb.set_trace()