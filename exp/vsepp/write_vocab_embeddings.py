import argparse
from tqdm import tqdm
import numpy as np

import utils.io as io
from utils.argparse_utils import manage_required_args


parser = argparse.ArgumentParser()
parser.add_argument(
    '--vocab_pkl',
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
    '--vsepp_word_vecs_npy', 
    type=str, 
    default=None)


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def main():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'vocab_pkl',
            'visual_word_vecs_h5py',
            'visual_word_vecs_idx_json',
            'vsepp_word_vecs_npy'],
        optional_args=['visual_words_json'])

    vocab = io.load_pickle_object(args.vocab_pkl,compress=False)
    visual_word_vecs = io.load_h5py_object(
        args.visual_word_vecs_h5py)['embeddings'][()]
    visual_word_vecs_idx = io.load_json_object(args.visual_word_vecs_idx_json)

    if args.visual_words_json is None:
        visual_words = set()
    else:
        visual_words = set(io.load_json_object(args.visual_words_json))
    
    vec_dim = visual_word_vecs.shape[1]
    vsepp_word_vecs = np.zeros([len(vocab),vec_dim])

    count = 0
    visual_count = 0
    for word,vsepp_idx in vocab.word2idx.items():
        if word in visual_word_vecs_idx:
            visual_idx = visual_word_vecs_idx[word]
            vec = visual_word_vecs[visual_idx]
            vsepp_word_vecs[vsepp_idx] = vec
            count += 1
            if word in visual_words:
                visual_count += 1

    print('Fraction available in glove: ',count/len(vocab))
    print('Fraction available in visual: ',visual_count/len(vocab))

    np.save(args.vsepp_word_vecs_npy,vsepp_word_vecs)

if __name__=='__main__':
    main()