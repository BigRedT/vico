import os
import h5py
from tqdm import tqdm
import numpy as np

import utils.io as io
from utils.constants import save_constants


def main(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    save_constants(
        {'exp': exp_const, 'data': data_const},
        exp_const.exp_dir)
    
    print('Loading glove embeddings ...')
    glove_idx = io.load_json_object(data_const.glove_idx)
    glove_h5py = h5py.File(
        data_const.glove_h5py,
        'r')
    glove_embeddings = glove_h5py['embeddings'][()]
    num_glove_words, glove_dim = glove_embeddings.shape
    print('-'*80)
    print(f'number of glove words: {num_glove_words}')
    print(f'glove dim: {glove_dim}')
    print('-'*80)
 
    random = 2*(np.random.rand(num_glove_words,exp_const.random_dim)-0.5)
    word_vecs = np.concatenate((glove_embeddings,random),1)
    word_vec_dim = glove_dim + exp_const.random_dim
    word_vecs_h5py =  h5py.File(
        os.path.join(exp_const.exp_dir,'glove_random_word_vecs.h5py'),
        'w') 
    word_vecs_h5py.create_dataset(
        'embeddings',
        data=word_vecs,
        chunks=(1,word_vec_dim))

