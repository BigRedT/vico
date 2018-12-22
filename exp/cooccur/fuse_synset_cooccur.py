import os
import copy
import itertools
from tqdm import tqdm
from nltk.corpus import wordnet as wn

import utils.io as io
from utils.constants import save_constants


def main(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    save_constants(
        {'exp': exp_const,'data': data_const},
        exp_const.exp_dir)

    imagenet_synset_cooccur = io.load_json_object(
        data_const.imagenet_synset_cooccur_json)

    genome_synset_cooccur = io.load_json_object(
        data_const.genome_synset_cooccur_json)

    fused_synset_cooccur = copy.deepcopy(imagenet_synset_cooccur)
    for wnid1,context in tqdm(genome_synset_cooccur.items()):
        if wnid1 not in fused_synset_cooccur:
            fused_synset_cooccur[wnid1] = {}

        for wnid2,count in context.items():
            if wnid2 not in fused_synset_cooccur[wnid1]:
                fused_synset_cooccur[wnid1][wnid2] = 0

            fused_synset_cooccur[wnid1][wnid2] += count

    fused_synset_cooccur_json = os.path.join(
        exp_const.exp_dir,
        'fused_synset_cooccur.json')
    io.dump_json_object(fused_synset_cooccur,fused_synset_cooccur_json)

    print('Checking symmetry and self constraint in synset cooccur ...')
    sym_err_msg = 'Word cooccurence not symmetric ...'
    self_err_msg = 'Self constraints violated ...'
    for wnid1, context in tqdm(fused_synset_cooccur.items()):
        for wnid2, count in context.items():
            sym_err_msg = f'Word cooccurence not symmetric ({wnid1} / {wnid2})'
            self_err_msg = f'Self constraints violated ({wnid1} / {wnid2})'
            assert(fused_synset_cooccur[wnid2][wnid1]==count), err_msg
            assert(fused_synset_cooccur[wnid1][wnid1]>=count), self_err_msg
            assert(fused_synset_cooccur[wnid2][wnid2]>=count), self_err_msg
    
    print('Constraints satisfied')

    print('Num imagenet synsets: ',len(imagenet_synset_cooccur))
    print('Num genome synsets: ',len(genome_synset_cooccur))
    print('Num fused synsets: ',len(fused_synset_cooccur))