import argparse
from tqdm import tqdm
import numpy as np

import utils.io as io
from utils.argparse_utils import manage_required_args


parser = argparse.ArgumentParser()
parser.add_argument(
    '--cocotalk_json',
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
    '--cocotalk_word_vecs_npy', 
    type=str, 
    default=None)

def main():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'cocotalk_json',
            'visual_word_vecs_h5py',
            'visual_word_vecs_idx_json',
            'cocotalk_word_vecs_npy'],
        optional_args=['visual_words_json'])

    cocotalk = io.load_json_object(args.cocotalk_json)
    ix_to_word = cocotalk['ix_to_word']
    word_to_idx = {v:int(k) for k,v in ix_to_word.items()}

    visual_word_vecs = io.load_h5py_object(
        args.visual_word_vecs_h5py)['embeddings'][()]
    visual_word_vecs_idx = io.load_json_object(args.visual_word_vecs_idx_json)

    if args.visual_words_json is None:
        visual_words = set()
    else:
        visual_words = set(io.load_json_object(args.visual_words_json))

    vec_dim = visual_word_vecs.shape[1]
    cocotalk_word_vecs = np.zeros([len(word_to_idx)+1,vec_dim]) # because word_to_idx doesn't have idx 0

    count = 0
    visual_count = 0
    for word,cocotalk_idx in word_to_idx.items():
        if word in visual_word_vecs_idx:
            visual_idx = visual_word_vecs_idx[word]
            vec = visual_word_vecs[visual_idx]
            cocotalk_word_vecs[cocotalk_idx] = vec
            count += 1
            if word in visual_words:
                visual_count += 1

    print('Fraction available in glove: ',
        count/len(word_to_idx))
    print('Fraction available in visual: ',
        visual_count/len(word_to_idx))

    np.save(args.cocotalk_word_vecs_npy,cocotalk_word_vecs)
    
    
if __name__== '__main__':
    main()