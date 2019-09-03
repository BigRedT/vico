import os
import h5py
import argparse
from tqdm import tqdm
import numpy as np

import utils.io as io

parser = argparse.ArgumentParser()
parser.add_argument(
    '--glove_txt', 
    type=str, 
    default=os.path.join(
        os.getcwd(),
        'symlinks/data/glove/glove.6B/glove.6B.100d.txt'), 
    help='Path to the txt file containing glove embeddings')
parser.add_argument(
    '--out_dir',
    type=str,
    default=os.path.join(os.getcwd(),'symlinks/data/glove/proc'),
    help='Path to output directory')
parser.add_argument(
    '--name',
    type=str,
    default='glove_6B_100d',
    help='Path to output directory')

def main():
    args = parser.parse_args()

    io.mkdir_if_not_exists(args.out_dir)

    print(f'Reading {args.glove_txt} ...')
    f = open(args.glove_txt,'r')
    embeddings = []
    word_to_idx = {}
    for i,line in enumerate(tqdm(f)):
        splitLine = line.split()
        word = splitLine[0]
        embeddings.append(np.array([float(val) for val in splitLine[1:]]))
        word_to_idx[word] = i
    embeddings = np.stack(embeddings)
    print('Embedding matrix shape:',embeddings.shape)
    mean = np.mean(embeddings,axis=0)
    std = np.std(embeddings,axis=0)
    min_ = np.min(embeddings,axis=0)
    max_ = np.max(embeddings,axis=0)

    print(f'Creating glove.h5py ...')
    glove_h5py = h5py.File(
        os.path.join(
            args.out_dir,
            f'{args.name}.h5py'),
        'w')
    glove_h5py.create_dataset('embeddings',data=embeddings)
    glove_h5py.create_dataset('mean',data=mean)
    glove_h5py.create_dataset('std',data=std)
    glove_h5py.create_dataset('min',data=min_)
    glove_h5py.create_dataset('max',data=max_)
    glove_h5py.close()

    print(f'Saving glove_word_to_idx.json ...')
    io.dump_json_object(
        word_to_idx,
        os.path.join(args.out_dir,f'{args.name}_word_to_idx.json'))

    print(f'Vocab size: {len(word_to_idx)}')
    

if __name__ == '__main__':
    main()

