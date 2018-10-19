import os
import h5py
import numpy as np
from tqdm import tqdm

import utils.io as io

def main():
    embeddings_h5py = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/' + \
        'normalized_resnet_embeddings_recon_loss_trained_on_google/' + \
        'concat_glove_and_visual/visual_word_vecs.h5py')
    word_to_idx_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/' + \
        'normalized_resnet_embeddings_recon_loss_trained_on_google/' + \
        'concat_glove_and_visual/visual_word_vecs_idx.json')
    semaleval_all_words_json = os.path.join(
        os.getcwd(),
        'symlinks/data/semeval_2018_10/proc/all_words.json')
    
    print('Reading embeddings ...')
    embeddings = h5py.File(embeddings_h5py,'r')['embeddings'][()]
    mean_embedding = np.mean(embeddings,0)
    word_to_idx = io.load_json_object(word_to_idx_json)
    words_to_select = list(
        io.load_json_object(semaleval_all_words_json).keys()) 

    print('Selecting subset ...')
    subset_embeddings = np.zeros([len(words_to_select),embeddings.shape[1]])
    subset_word_to_idx = {}
    count = 0
    for i,word in enumerate(tqdm(words_to_select)):
        subset_word_to_idx[word] = i
        if word not in word_to_idx:
            count += 1
            subset_embeddings[i] = mean_embedding
        else:
            idx = word_to_idx[word]
            subset_embeddings[i] = embeddings[idx]
    print(count)

    print('Saving selected subset embeddings ...')
    subset_visual_word_vecs_h5py_filename = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/' + \
        'normalized_resnet_embeddings_recon_loss_trained_on_google/' + \
        'concat_glove_and_visual/subset_visual_word_vecs.h5py')
    subset_embeddings_h5py = h5py.File(
        subset_visual_word_vecs_h5py_filename,
        'w')
    subset_embeddings_h5py.create_dataset(
        'embeddings',
        data=subset_embeddings,
        chunks=(1,subset_embeddings.shape[1]))
    
    subset_word_to_idx_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/' + \
        'normalized_resnet_embeddings_recon_loss_trained_on_google/' + \
        'concat_glove_and_visual/subset_visual_word_vecs_idx.json')
    io.dump_json_object(subset_word_to_idx,subset_word_to_idx_json)


if __name__=='__main__':
    main()

    
    