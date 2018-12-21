import os
import copy
import itertools
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from torch.utils.data import DataLoader

import utils.io as io
from utils.constants import save_constants
from .dataset import ImagenetNoImgsDataset


def wnid_offset_to_synset(offset):
    synset = wn.synset_from_pos_and_offset(offset[0],int(offset[1:])).name()
    return synset


def create_gt_synset_cooccur(exp_const,dataloader):
    print('Creating cooccur ...')
    cooccur = {}
    for data in tqdm(dataloader):
        on_wnids = data['on_wnids']
        for b in range(len(on_wnids)):
            for wnid1,wnid2 in itertools.product(on_wnids[b],on_wnids[b]):
                if wnid1 not in cooccur:
                    cooccur[wnid1] = {}
                
                if wnid2 not in cooccur[wnid1]:
                    cooccur[wnid1][wnid2] = 0

                cooccur[wnid1][wnid2] += 1

    print('Creating offsets to synsets dict ...')
    offset_to_synset = {}
    for offset in tqdm(cooccur.keys()):
        offset_to_synset[offset] = wnid_offset_to_synset(offset)

    print('Replacing offset by synset in cooccur ...')
    synset_cooccur = {}
    for offset1 in tqdm(cooccur.keys()):
        synset1 = offset_to_synset[offset1]
        
        context = {}
        for offset2,count in cooccur[offset1].items():
            synset2 = offset_to_synset[offset2]
            context[synset2] = count
        
        synset_cooccur[synset1] = context
        
    synset_cooccur_json = os.path.join(exp_const.exp_dir,'synset_cooccur.json')
    io.dump_json_object(synset_cooccur,synset_cooccur_json)


def main(exp_const,data_const):
    print(f'Creating directory {exp_const.exp_dir} ...')
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)

    print('Saving constants ...')
    save_constants(
        {'exp': exp_const,'data': data_const},
        exp_const.exp_dir)

    print('Creating dataloader ...')
    data_const = copy.deepcopy(data_const)
    dataset = ImagenetNoImgsDataset(data_const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=False,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    create_gt_synset_cooccur(exp_const,dataloader)
    