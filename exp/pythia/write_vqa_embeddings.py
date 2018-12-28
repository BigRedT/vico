import argparse
from tqdm import tqdm
import numpy as np

import utils.io as io
from utils.argparse_utils import manage_required_args

parser = argparse.ArgumentParser()
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
    '--vqa_vocab_txt', 
    type=str, 
    default=None)
parser.add_argument(
    '--vqa_word_vecs_npy', 
    type=str, 
    default=None)
parser.add_argument(
    '--glove_dim', 
    type=int, 
    default=300)


def main():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'visual_word_vecs_h5py',
            'visual_word_vecs_idx_json',
            'visual_words_json',
            'vqa_vocab_txt',
            'vqa_word_vecs_npy'])

    visual_word_vecs = io.load_h5py_object(
        args.visual_word_vecs_h5py)['embeddings'][()]
    visual_word_vecs_idx = io.load_json_object(
        args.visual_word_vecs_idx_json)
    visual_words = set(io.load_json_object(args.visual_words_json))

    # Read vqa_vocab_txt file
    vqa_vocab = []
    for line in tqdm(open(args.vqa_vocab_txt,'r')):
        vqa_vocab.append(line[:-1])

    num_vqa_words = len(vqa_vocab)
    word_vec_dim = visual_word_vecs.shape[1]
    vqa_word_vecs = np.zeros([num_vqa_words,word_vec_dim])
    count = 0
    visual_count = 0

    for i,word in enumerate(tqdm(vqa_vocab)):
        if word in visual_word_vecs_idx:
            idx = visual_word_vecs_idx[word]
            vqa_word_vecs[i] = visual_word_vecs[idx]
            count += 1
            if word in visual_words:
                visual_count += 1

    print('Fraction available in glove: ',count/len(vqa_vocab))
    print('Fraction available in visual: ',visual_count/len(vqa_vocab))
    
    print(f'Writing vqa word vectors to {args.vqa_word_vecs_npy} ...')
    np.save(args.vqa_word_vecs_npy,vqa_word_vecs)


if __name__=='__main__':
    main()
    
