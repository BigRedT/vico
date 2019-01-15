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
        

class MultiSenseCooccurDataset(Dataset):
    def __init__(self,const):
        super(MultiSenseCooccurDataset,self).__init__()
        self.const = copy.deepcopy(const)
        print('Reading csv in pandas dataframe ...')
        self.df = pd.read_csv(self.const.cooccur_csv)
        self.column_names = self.df.columns.values.tolist()
        self.cooccur_types = self.column_names[2:]
        self.normalize_counts()
        self.median_counts = self.get_median_counts()
        self.words = sorted([
            str(s) for s in list(set(self.df.word1.values.tolist()))])
        self.word_to_idx = {word:idx for idx,word in enumerate(self.words)}
        
    def normalize_counts(self):
        for cooccur_type in self.cooccur_types:
            df_col = self.df[cooccur_type]
            df_select = df_col[df_col>0.5]
            mi = np.min(df_select)
            ma = np.max(df_select)
            self.df.loc[df_col>0.5,cooccur_type] = df_select + 0 #((df_select-mi)*1/(ma+1e-6)) + 1

    def get_median_counts(self):
        median_counts = {}
        for cooccur_type in self.cooccur_types:
            x = self.df[cooccur_type]
            median_counts[cooccur_type] = np.median(x[x>0.5])
            print(np.max(x[x>0.5]))

        print(median_counts)
        return median_counts

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self,i):
        row = self.df.iloc[i]
        to_return = {}
        for name in self.column_names:
            to_return[name] = row[name]
        
        to_return['word1'] = str(to_return['word1'])
        to_return['word2'] = str(to_return['word2'])

        if to_return['word1'] not in self.word_to_idx:
            return None

        if to_return['word2'] not in self.word_to_idx:
            return None

        to_return['idx1'] = self.word_to_idx[to_return['word1']]
        to_return['idx2'] = self.word_to_idx[to_return['word2']]

        if to_return['idx1']==to_return['idx2']:
            return None

        for name in self.cooccur_types:
            mask_value = True
            if to_return[name] < 1:
                mask_value = False
            
            to_return[name + '_mask'] = mask_value
        
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

    def get_batch_by_type(self,batch,cooccur_type):
        if batch is None:
            return None
            
        mask = batch[cooccur_type + '_mask']
        if not any(mask):
            return None

        to_return = {}
        for name in ['word1','word2','idx1','idx2',cooccur_type]:
            if name==cooccur_type:
                name_ = 'x'
            else:
                name_ = name

            to_return[name_] = [
                v for v,m in zip(batch[name],mask) if m==True]

        to_return['x'] = [float(v) for v in to_return['x']]

        return to_return

if __name__=='__main__':
    const = MultiSenseCooccurDatasetConstants()
    const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt/merged_cooccur.csv')

    dataset = MultiSenseCooccurDataset(const)    
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn)
    for data_ in dataloader:
        data = dataset.get_batch_by_type(data_,'context')
        if data is not None:
            [print(p) for p in zip(
                data['word1'],data['word2'],data['x'])]
        import pdb; pdb.set_trace()