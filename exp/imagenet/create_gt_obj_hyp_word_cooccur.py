import os
import re
import copy
import itertools
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from torch.utils.data import DataLoader

import utils.io as io
from utils.constants import save_constants
from .dataset import ImagenetNoImgsDataset


def wnid_offset_to_word(offset):
    synset = wn.synset_from_pos_and_offset(offset[0],int(offset[1:])).name()
    return synset.split('.')[0]


def on_wnids_to_words(on_wnids,mem):
    words = set()
    for wnid in on_wnids:
        if wnid in mem:
            proc_word = mem[wnid]
        else:
            raw_word = wnid_offset_to_word(wnid)
            proc_word = re.sub("-|_",' ',raw_word)
            proc_word = re.sub("\'s",'',proc_word)
            mem[wnid] = proc_word

        for word in proc_word.split(' '):
            if word=='':
                continue
            words.add(word)
        
    return words


def create_gt_synset_cooccur(exp_const,dataloader):
    print('Creating cooccur ...')
    cooccur = {}
    mem = {}
    for data in tqdm(dataloader):
        on_wnids = data['on_wnids']
        for b in range(len(on_wnids)):
            words = on_wnids_to_words(on_wnids[b],mem)
            for word1 in set(words):
                for word2 in set(words):
                    if word1 not in cooccur:
                        cooccur[word1] = {}
                    
                    if word2 not in cooccur[word1]:
                        cooccur[word1][word2] = 0

                    cooccur[word1][word2] += 1
        
    word_cooccur_json = os.path.join(exp_const.exp_dir,'word_cooccur.json')
    io.dump_json_object(cooccur,word_cooccur_json)

    print('Checking symmetry constraint in word cooccur ...')
    for wnid1, context in tqdm(cooccur.items()):
        for wnid2, count in context.items():
            sym_err_msg = f'Word cooccurence not symmetric ({wnid1} / {wnid2})'
            assert(cooccur[wnid2][wnid1]==count), sym_err_msg

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
    dataset = ImagenetNoImgsDataset(data_const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=False,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    create_gt_synset_cooccur(exp_const,dataloader)
    