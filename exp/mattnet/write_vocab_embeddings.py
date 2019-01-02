import argparse
from tqdm import tqdm
import numpy as np

import utils.io as io
from utils.argparse_utils import manage_required_args


parser = argparse.ArgumentParser()
parser.add_argument(
    '--mattnet_data_json',
    type=str)
parser.add_argument(
    '--visual_word_vecs_h5py', 
    type=str, 
    default=None)
parser.add_argument(
    '--visual_word_vecs_idx_json', 
    type=str, 
    default=None)
parser.add_argument(
    '--visual_words_json', 
    type=str, 
    default=None)
parser.add_argument(
    '--mattnet_word_vecs_npy', 
    type=str, 
    default=None)


def main():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'mattnet_data_json',
            'visual_word_vecs_h5py',
            'visual_word_vecs_idx_json',
            'mattnet_word_vecs_npy'],
        optional_args=['visual_words_json'])

    mattnet_data = io.load_json_object(args.mattnet_data_json)
    mattnet_word_to_idx = mattnet_data['word_to_ix']

    visual_word_vecs = io.load_h5py_object(
        args.visual_word_vecs_h5py)['embeddings'][()]
    visual_word_vecs_idx = io.load_json_object(args.visual_word_vecs_idx_json)

    if args.visual_words_json is None:
        visual_words = set()
    else:
        visual_words = set(io.load_json_object(args.visual_words_json))
    
    vec_dim = visual_word_vecs.shape[1]
    mattnet_word_vecs = np.zeros([len(mattnet_word_to_idx),vec_dim])

    count = 0
    visual_count = 0
    for word,mattnet_idx in mattnet_word_to_idx.items():
        if word in visual_word_vecs_idx:
            visual_idx = visual_word_vecs_idx[word]
            vec = visual_word_vecs[visual_idx]
            mattnet_word_vecs[mattnet_idx] = vec
            count += 1
            if word in visual_words:
                visual_count += 1

    print('Fraction available in glove: ',count/len(mattnet_word_to_idx))
    print('Fraction available in visual: ',visual_count/len(mattnet_word_to_idx))

    np.save(args.mattnet_word_vecs_npy,mattnet_word_vecs)
    

if __name__=='__main__':
    main()