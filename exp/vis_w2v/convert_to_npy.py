# Script to obtain the vocabulary from the embeddings file
import sys
import time
import os
from tqdm import tqdm
import numpy as np

import utils.io as io


def main(embedPath,outdir,vocab_json,embed_type):
    io.mkdir_if_not_exists(outdir)

    vocab = io.load_json_object(vocab_json)

    with open(embedPath, 'r', encoding='latin') as fileId:
        # Read only the word, ignore feature vector
        lines = []
        for line in tqdm(fileId.readlines()):
            lines.append(line.split(' ', 1))
            #import pdb; pdb.set_trace()
        #lines = [line.split(' ', 1) for line in fileId.readlines()]; #[0]

    #vocab_size = int(lines[0][0])
    vocab_size = len(vocab)
    dim = int(lines[0][1][:-1])
    print(vocab_size,dim)
    embed = np.zeros([vocab_size,dim])
    word_to_idx = {}
    count = 0
    for line in tqdm(lines[1:]):
        word = str(line[0])#.lower()
        if word not in vocab:
            continue
        
        vec = line[1] # space separated string of numbers with '\n' at the end
        if embed_type=='word2vec_wiki' or embed_type=='visual_word2vec_wiki':
            vec = vec[:-1]
            vec = vec.split(' ')
        else:
            vec = vec.split(' ')[:-1] # get rid of the '\n'
        
        count = vocab[word]
        word_to_idx[word] = count
        embed[count] = [float(s) for s in vec]
        #count+=1

    import pdb; pdb.set_trace()
    embed_npy = os.path.join(outdir,'visual_embeddings.npy')
    np.save(embed_npy,embed)

    word_to_idx_json = os.path.join(outdir,'word_to_idx.json')
    io.dump_json_object(word_to_idx,word_to_idx_json)


if __name__ == '__main__':
    embedPath = sys.argv[1];
    outdir = sys.argv[2]
    vocab_json = sys.argv[3]
    embed_type = sys.argv[4]

    if not os.path.exists(embedPath):
        print(f'File at {embedPath} does not exist!')
    else:
        main(embedPath,outdir,vocab_json,embed_type);