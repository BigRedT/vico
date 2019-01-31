import os
import copy
import itertools
from tqdm import tqdm
import pandas as pd

import utils.io as io
from utils.constants import save_constants


def main(exp_const,data_const):
    print(f'Creating directory {exp_const.exp_dir} ...')
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)

    print('Saving constants ...')
    save_constants(
        {'exp': exp_const,'data': data_const},
        exp_const.exp_dir)

    print('Loading cooccur ...')
    num_cooccur_types = len(data_const.cooccur_paths)
    merged_cooccur = {}
    for i, (cooccur_type,cooccur_json) in \
        enumerate(data_const.cooccur_paths.items()):
        
        print(f'    Merging {cooccur_type} ...')
        cooccur = io.load_json_object(cooccur_json)
        for word1,context in tqdm(cooccur.items()):
            if word1 not in merged_cooccur:
                merged_cooccur[word1] = {}
            
            for word2,count in context.items():    
                if word2 not in merged_cooccur[word1]:
                    merged_cooccur[word1][word2] = [0]*num_cooccur_types

                merged_cooccur[word1][word2][i] += count

    if exp_const.normalize==True:
        print('Normalizing by self counts ...')
        for word1,context in tqdm(merged_cooccur.items()):
            norm_counts = merged_cooccur[word1][word1]

            for word2,counts in context.items():
                if word2==word1:
                    continue

                for i, norm_count in enumerate(norm_counts):
                    counts[i] = counts[i] / (norm_count + 1e-6)
                    counts[i] = min(1,counts[i])

            merged_cooccur[word1][word1] = [1]*len(norm_counts)
            
    pandas_cols = {
        'word1': [],
        'word2': [],
    }
    for cooccur_type in data_const.cooccur_paths.keys():
        pandas_cols[cooccur_type] = []

    print('Creating pandas columns ...')
    for word1, context in tqdm(merged_cooccur.items()):
        for word2, counts in context.items():
            pandas_cols['word1'].append(word1)
            pandas_cols['word2'].append(word2)
            for i,cooccur_type in enumerate(data_const.cooccur_paths.keys()):
                pandas_cols[cooccur_type].append(counts[i])

    pandas_cols['word1'] = pd.Categorical(pandas_cols['word1'])
    pandas_cols['word2'] = pd.Categorical(pandas_cols['word2'])

    for cooccur_type in data_const.cooccur_paths.keys():
        pandas_cols[cooccur_type] = pd.Series(pandas_cols[cooccur_type])

    df = pd.DataFrame(pandas_cols)

    print('Saving DataFrame to csv ...')
    df.to_csv(data_const.merged_cooccur_csv,index=False)
    
    
