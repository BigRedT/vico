import os
import copy
import numpy as np
from tqdm import tqdm

import utils.io as io


def main(exp_const,data_const):
    word_cooccur = io.load_json_object(data_const.word_cooccur_json)
    
    print('Gather word counts ...')
    word_counts = {}
    for word in word_cooccur.keys():
        word_counts[word] = word_cooccur[word][word]

    sqrt_word_counts = {k:np.sqrt(v) for k,v in word_counts.items()}

    print('Scale cooccur ...')
    proc_word_cooccur = copy.deepcopy(word_cooccur)
    for word1, context in tqdm(proc_word_cooccur.items()):
        n1 = sqrt_word_counts[word1]
        for word2, count in context.items():
            n2 = sqrt_word_counts[word2]
            context[word2] = [count,round(count / (n1*n1+1e-6),4)]
            
    print('Filter ...')
    for word1,context in tqdm(proc_word_cooccur.items()):
        context_ = {w:c for w,c in context.items() if c[1] > 0.01}
        proc_word_cooccur[word1] = context_
            
    io.dump_json_object(proc_word_cooccur,data_const.proc_word_cooccur_json)
    