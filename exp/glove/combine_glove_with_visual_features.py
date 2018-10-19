import os
import h5py
from tqdm import tqdm
import numpy as np

import utils.io as io
from utils.constants import save_constants


def compute_norm(x):
    return np.linalg.norm(x,2)


def normalize(x):
    return x / (1e-6 + compute_norm(x))


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

    print('Loading visual features ...')
    visual_features_idx = io.load_json_object(data_const.visual_features_idx)
    visual_features_h5py = h5py.File(
        data_const.visual_features_h5py,
        'r')
    visual_features = visual_features_h5py['features'][()]
    num_visual_features, visual_features_dim = visual_features.shape
    print('-'*80)
    print(f'number of visual features: {num_visual_features}')
    print(f'visual feature dim: {visual_features_dim}')
    print('-'*80)

    print('Combining glove with visual features ...')
    visual_word_vecs_idx_json = os.path.join(
        exp_const.exp_dir,
        'visual_word_vecs_idx.json')
    io.dump_json_object(glove_idx,visual_word_vecs_idx_json)
    visual_word_vecs_h5py =  h5py.File(
        os.path.join(exp_const.exp_dir,'visual_word_vecs.h5py'),
        'w') 
    visual_word_vec_dim = glove_dim + visual_features_dim
    visual_word_vecs = np.zeros([num_glove_words,visual_word_vec_dim])
    mean_visual_feature = visual_features_h5py['mean'][()]
    for word in tqdm(glove_idx.keys()):
        glove_id = glove_idx[word]
        glove_vec = glove_embeddings[glove_id]
        if word in visual_features_idx:
            feature_id = visual_features_idx[word]
            feature = visual_features[feature_id]
        else:
            feature = mean_visual_feature
        visual_word_vec = np.concatenate((
            glove_vec,
            (feature-mean_visual_feature)))
        # visual_word_vec = np.concatenate((
        #     normalize(glove_vec),
        #     normalize(feature)))
        visual_word_vecs[glove_id] = visual_word_vec
    
    visual_word_vecs_h5py.create_dataset(
        'embeddings',
        data=visual_word_vecs,
        chunks=(1,visual_word_vec_dim))
    visual_word_vecs_h5py.close()

