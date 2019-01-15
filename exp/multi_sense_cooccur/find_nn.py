import os
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import utils.io as io
from utils.html_writer import HtmlWriter


def main(exp_const,data_const):
    print('Loading embeddings ...')
    embeddings = np.load(data_const.embeddings_npy)
    word_to_idx = io.load_json_object(data_const.word_to_idx_json)
    idx_to_word = {v:k for k,v in word_to_idx.items()}
    df = pd.read_csv(data_const.cooccur_csv)

    for k,cooccur_type in enumerate(exp_const.cooccur_types):
        print(f'Finding knn for {cooccur_type} ...')
        print('Loading xform ...')
        xform = np.load(os.path.join(
            exp_const.exp_dir,
            f'xform_{cooccur_type}.npy'))

        print('Selecting words ...')
        sorted_df = df.sort_values(by=cooccur_type,ascending=False)
        sorted_df = sorted_df[:exp_const.thresh[k]]
        select_words_ = list(set(
            sorted_df.word1.values.tolist() + \
            sorted_df.word2.values.tolist()))
        select_words = []
        for word in select_words_:
            if word in word_to_idx:
                select_words.append(word)
        print('Number of selected words: ',len(select_words))
        select_word_idxs = [word_to_idx[word] for word in select_words]
        select_joint_embeddings = embeddings[select_word_idxs]
        select_embeddings = np.matmul(
            select_joint_embeddings,
            np.transpose(xform))
        if exp_const.use_cosine==True:
            select_joint_embeddings = select_joint_embeddings / \
                (1e-6+np.linalg.norm(select_joint_embeddings,2,1,keepdims=True))
            select_embeddings = select_embeddings / \
                (1e-6+np.linalg.norm(select_embeddings,2,1,keepdims=True))

        print('Computing similarity using transformed embeddings ...')
        sim = np.matmul(select_embeddings,np.transpose(select_embeddings))
        print('Sim Range: {:.4f} - {:.4f}'.format(np.min(sim),np.max(sim)))
        nn_idx = np.argsort(sim,1)[:,-2:-exp_const.num_nbrs-2:-1]

        print('Computing similarity using joint embeddings ...')
        joint_sim = np.matmul(
            select_joint_embeddings,
            np.transpose(select_joint_embeddings))
        print('Sim Range: {:.4f} - {:.4f}'.format(
            np.min(joint_sim),np.max(joint_sim)))
        joint_nn_idx = np.argsort(joint_sim,1)[:,-2:-exp_const.num_nbrs-2:-1]
        
        nbrs = {}
        for i in tqdm(range(sim.shape[0])):
            query_word = select_words[i]
            nbrs[query_word] = {
                'words': [select_words[k] for k in nn_idx[i]],
                'joint_words': [select_words[k] for k in joint_nn_idx[i]],
                'sim': [sim[i,k] for k in nn_idx[i]],
                'joint_sim': [joint_sim[i,k] for k in nn_idx[i]]
            }
            
        cooccur_nbrs = {}
        for i in tqdm(range(sim.shape[0])):
            query_word = select_words[i]
            words = []
            sim = []
            rows = sorted_df[sorted_df.word1==query_word]
            for j in range(exp_const.num_nbrs):
                if j >= len(rows):
                    break

                row = rows.iloc[j]
                
                words.append(str(row['word2']))
                sim.append(np.log(row[cooccur_type]+1e-6))

            cooccur_nbrs[query_word] = {
                'words': words,
                'sim': sim,
            }

        cosine_suffix = ''
        if exp_const.use_cosine==True:
            cosine_suffix = '_cosine'
        html_filename = os.path.join(
            exp_const.exp_dir,
            f'visual_embeddings_knn{cosine_suffix}_{cooccur_type}.html')
        html_writer = HtmlWriter(html_filename)
        col_dict = {
            0: 'Query',
            1: 'Nbrs'
        }
        html_writer.add_element(col_dict)
        for query_word in sorted(nbrs.keys()):
            # Add cooccurrence nbrs
            nbr_str = ''
            for word,score in zip(
                cooccur_nbrs[query_word]['words'],
                cooccur_nbrs[query_word]['sim']):
                nbr_str += '{}({:.2f})'.format(word,score)
                nbr_str += '&nbsp;'*4

            col_dict = {
                0: '\"{}\"'.format(query_word),
                1: nbr_str
            }
            html_writer.add_element(col_dict)

            # Add embedding nbrs
            nbr_str = ''
            for word,score in zip(
                nbrs[query_word]['words'],
                nbrs[query_word]['sim']):
                nbr_str += '{}({:.2f})'.format(word,score)
                nbr_str += '&nbsp;'*4

            intersection = set(nbrs[query_word]['words']).intersection(
                set(cooccur_nbrs[query_word]['words']))
            overlap = len(intersection) / \
                (len(cooccur_nbrs[query_word]['words'])+1e-6)
            col_dict = {
                0: f'xform ({round(overlap,2)})',
                1: nbr_str
            }
            html_writer.add_element(col_dict)

            # Add joint embedding nbrs
            nbr_str = ''
            for word,score in zip(
                nbrs[query_word]['joint_words'],
                nbrs[query_word]['joint_sim']):
                nbr_str += '{}({:.2f})'.format(word,score)
                nbr_str += '&nbsp;'*4

            intersection = set(nbrs[query_word]['joint_words']).intersection(
                set(cooccur_nbrs[query_word]['words']))
            overlap = len(intersection) / \
                (len(cooccur_nbrs[query_word]['words'])+1e-6)
            col_dict = {
                0: f'joint ({round(overlap,2)})',
                1: nbr_str
            }
            html_writer.add_element(col_dict)

            html_writer.add_element({0:'&nbsp;',1:'&nbsp;'})

        html_writer.close()
