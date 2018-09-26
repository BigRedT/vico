import os
import h5py
import numpy as np

import utils.io as io
from data.glove.constants import GloveConstantsFactory


def main():
    glove_const = GloveConstantsFactory.create()
    glove_embeddings_h5py = h5py.File(
        glove_const.embeddings_h5py,
        'r')
    glove_embeddings = glove_embeddings_h5py['embeddings'][()]
    mean = np.mean(glove_embeddings)
    std = np.std(glove_embeddings)
    min_ = np.min(glove_embeddings)
    max_ = np.max(glove_embeddings)
    num_words, dim = glove_embeddings.shape

    # Create random normal embeddings
    random_normal_embeddings_h5py = h5py.File(
        os.path.join(glove_const.proc_dir,'random_normal_embeddings.h5py'),
        'w')
    random_normal_embeddings = np.random.normal(
        loc=mean,
        scale=std,
        size=(num_words,dim))
    random_normal_embeddings = np.minimum(
        random_normal_embeddings,
        max_)
    random_normal_embeddings = np.maximum(
        random_normal_embeddings,
        min_)
    random_normal_embeddings_h5py.create_dataset(
        'embeddings',
        data=random_normal_embeddings)
    random_normal_embeddings_h5py.close()

    # Create random uniform embeddings
    random_uniform_embeddings_h5py = h5py.File(
        os.path.join(glove_const.proc_dir,'random_uniform_embeddings.h5py'),
        'w')
    random_uniform_embeddings = np.random.uniform(
        low=min_,
        high=max_,
        size=(num_words,dim))
    random_uniform_embeddings_h5py.create_dataset(
        'embeddings',
        data=random_uniform_embeddings)
    random_uniform_embeddings_h5py.close()

if __name__=='__main__':
    main()

    
