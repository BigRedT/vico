import numpy as np
import random

import utils.io as io
from utils.html_writer import HtmlWriter


def main(exp_const,data_const):
    print('Loading embeddings ...')
    embeddings = np.load(data_const.embeddings_npy)
    word_to_idx = io.load_json_object(data_const.word_to_idx_json)
    cooccur = io.load_json_object(data_const.cooccur_json)
    word_freq = {word: cooccur[word][word] for word in word_to_idx.keys()}

    print('Select words with freq > 1000 ...')
    select_words = [word for word in word_to_idx.keys() if \
        word_freq[word] > exp_const.min_freq]
    num_select_words = len(select_words)
    print('Num selected words: ',num_select_words)
    select_word_idxs = [word_to_idx[word] for word in select_words]
    select_embeddings = embeddings[select_word_idxs]
    # select_embeddings = select_embeddings / \
    #     (1e-6+np.linalg.norm(select_embeddings,2,1,keepdims=True))

    print('Computing similarity ...')
    # embeddings: num_classes x dim
    sim = np.matmul(select_embeddings,np.transpose(select_embeddings))
    print('Sim Range: {:.4f} - {:.4f}'.format(np.min(sim),np.max(sim)))
    nn_idx = np.argsort(sim,1)[:,-2:-exp_const.num_nbrs-2:-1]
    nbrs = {}
    for i in range(sim.shape[0]):
        query_word = select_words[i]
        nbrs[query_word] = {
            'words': [select_words[k] for k in nn_idx[i]],
            'sim': [sim[i,k] for k in nn_idx[i]]
        }

    print('Finding top cooccurring nbrs ...')
    select_cooccur = np.zeros([num_select_words,num_select_words])
    for i,word1 in enumerate(select_words):
        context = cooccur[word1]
        for j,word2 in enumerate(select_words):
            if word2 in context:
                select_cooccur[i,j] = context[word2]

    cooccur_nn_idx = np.argsort(select_cooccur,1)[:,-2:-exp_const.num_nbrs-2:-1]
    cooccur_nbrs = {}
    for i in range(select_cooccur.shape[0]):
        query_word = select_words[i]
        cooccur_nbrs[query_word] = {
            'words': [select_words[k] for k in cooccur_nn_idx[i]],
            'sim': [sim[i,k] for k in cooccur_nn_idx[i]]
        }


    html_writer = HtmlWriter(data_const.knn_html)
    col_dict = {
        0: 'Query',
        1: 'Nbrs'
    }
    html_writer.add_element(col_dict)
    for query_word in sorted(nbrs.keys()):
        # Add embedding nbrs
        nbr_str = ''
        for word,score in zip(nbrs[query_word]['words'],nbrs[query_word]['sim']):
            nbr_cooccur_freq = 0
            if word in cooccur[query_word]:
                nbr_cooccur_freq = cooccur[query_word][word]
            
            nbr_str += '{}({:.2f},{})'.format(
                word,
                score,
                nbr_cooccur_freq)
            nbr_str += '&nbsp;'*4

        col_dict = {
            0: '{}({})'.format(query_word,word_freq[query_word]),
            1: nbr_str
        }
        html_writer.add_element(col_dict)

        # Add cooccurrence nbrs
        nbr_str = ''
        for word,score in zip(
            cooccur_nbrs[query_word]['words'],
            cooccur_nbrs[query_word]['sim']):
            nbr_cooccur_freq = 0
            if word in cooccur[query_word]:
                nbr_cooccur_freq = cooccur[query_word][word]
            
            nbr_str += '{}({:.2f},{})'.format(
                word,
                score,
                nbr_cooccur_freq)
            nbr_str += '&nbsp;'*4

        intersection = set(nbrs[query_word]['words']).intersection(
            set(cooccur_nbrs[query_word]['words']))
        overlap = len(intersection) / \
            (len(cooccur_nbrs[query_word]['words'])+1e-6)
        col_dict = {
            0: round(overlap,2),
            1: nbr_str
        }
        html_writer.add_element(col_dict)

    html_writer.close()

