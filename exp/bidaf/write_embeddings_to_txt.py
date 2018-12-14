import os
import h5py
import subprocess
from tqdm import tqdm

import utils.io as io


def write_to_txt(outdir,embeddings_h5py,words_to_idx_json):
    embeddings = io.load_h5py_object(embeddings_h5py)['embeddings'][()].tolist()
    words_to_idx = io.load_json_object(words_to_idx_json)
    
    # create txt file
    embeddings_txt = os.path.join(outdir,'embeddings.txt')
    file = open(embeddings_txt,'w')
    for i, (word,idx) in enumerate(tqdm(words_to_idx.items())):
        line = word + ' ' + ' '.join(['{:.5f}'.format(v) for v in embeddings[idx]]) + '\n'
        file.write(line)
    file.close()
    
    # gzip txt file
    embeddings_txt_gz = os.path.join(outdir,'embeddings.txt.gz')
    gzip_cmd = f'gzip -c {embeddings_txt} > {embeddings_txt_gz}'
    subprocess.run(gzip_cmd,shell=True)


if __name__=='__main__':
    embeddings_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/combine_glove_visual_reps/concat_glove_visual_avg_reps')
    embeddings_h5py = os.path.join(embeddings_dir,'visual_word_vecs.h5py')
    words_to_idx_json = os.path.join(embeddings_dir,'visual_word_vecs_idx.json')
    outdir = os.path.join(
        os.getcwd(),
        'symlinks/exp/bidaf/concat_glove_visual_avg_reps')
    io.mkdir_if_not_exists(outdir,recursive=True)
    write_to_txt(outdir,embeddings_h5py,words_to_idx_json)
    