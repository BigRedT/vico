import os
import copy
import itertools
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from torch.utils.data import DataLoader

import utils.io as io
from utils.constants import save_constants
from .dataset import GenomeAttributesNoImgsDataset


def create_gt_synset_cooccur(exp_const,dataloader):
    print('Creating cooccur ...')
    cooccur = {}
    for data in tqdm(dataloader):
        B = len(data['object_synsets'])

        for b in range(B):
            object_wnids_set = set(data['object_synsets'][b])
            attribute_wnids_set = set(data['attribute_synsets'][b])
            for wnid1 in object_wnids_set:
                for wnid2 in attribute_wnids_set:
                    # Diagonals added later
                    if wnid1==wnid2:
                        continue

                    if wnid1 not in cooccur:
                        cooccur[wnid1] = {}
                    
                    if wnid2 not in cooccur[wnid1]:
                        cooccur[wnid1][wnid2] = 0

                    cooccur[wnid1][wnid2] += 1

                    if wnid2 not in cooccur:
                        cooccur[wnid2] = {}
                    
                    if wnid1 not in cooccur[wnid2]:
                        cooccur[wnid2][wnid1] = 0

                    cooccur[wnid2][wnid1] += 1

            # Add diagonals
            for wnid1 in object_wnids_set:
                if wnid1 not in cooccur:
                    cooccur[wnid1] = {}

                if wnid1 not in cooccur[wnid1]:
                    cooccur[wnid1][wnid1] = 0
                
                cooccur[wnid1][wnid1] += 1

            for wnid2 in attribute_wnids_set:
                if wnid2 not in cooccur:
                    cooccur[wnid2] = {}

                if wnid2 not in cooccur[wnid2]:
                    cooccur[wnid2][wnid2] = 0
                
                cooccur[wnid2][wnid2] += 1
        
    synset_cooccur_json = os.path.join(exp_const.exp_dir,'synset_cooccur.json')
    io.dump_json_object(cooccur,synset_cooccur_json)

    print('Checking symmetry and self constraint in synset cooccur ...')
    for wnid1, context in tqdm(cooccur.items()):
        for wnid2, count in context.items():
            sym_err_msg = f'Word cooccurence not symmetric ({wnid1} / {wnid2})'
            assert(cooccur[wnid2][wnid1]==count), err_msg

    print('Constraints satisfied')
    

def main(exp_const,data_const):
    print(f'Creating directory {exp_const.exp_dir} ...')
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)

    print('Saving constants ...')
    save_constants(
        {'exp': exp_const,'data': data_const},
        exp_const.exp_dir)

    print('Creating dataloader ...')
    data_const = copy.deepcopy(data_const)
    dataset = GenomeAttributesNoImgsDataset(data_const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets',
        'object_words','attribute_words',
        'attribute_labels_idxs'])
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=False,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    create_gt_synset_cooccur(exp_const,dataloader)
    