import os

import utils.io as io
from .constants import SquadConstants


def compute_overlap(squad_vocab,visual_vocab):
    intersection = squad_vocab.intersection(visual_vocab)
    frac = len(intersection) / (len(squad_vocab) +  1e-6)
    frac = round(frac*100,2)
    return frac


if __name__=='__main__':
    const = SquadConstants()
    squad_vocab_json = os.path.join(const.raw_dir,'dev_vocab.json')
    squad_vocab = io.load_json_object(squad_vocab_json)

    visual_vocab_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/combine_glove_visual_reps/' + \
        'concat_glove_visual_avg_reps/visual_words.json')
    visual_vocab = io.load_json_object(visual_vocab_json)

    glove_vocab_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/combine_glove_visual_reps/' + \
        'concat_glove_visual_avg_reps/visual_word_vecs_idx.json')
    glove_vocab = io.load_json_object(glove_vocab_json)

    visual_overlap = {}
    glove_overlap = {}
    for k,v in squad_vocab.items():
        visual_overlap[k] =  compute_overlap(set(v),set(visual_vocab))
        glove_overlap[k] = compute_overlap(set(v),set(glove_vocab.keys()))
    
    print('-'*80)
    print('Overlap with visual vocab')
    print('-'*80)
    print(visual_overlap)

    print('-'*80)
    print('Overlap with glove vocab')
    print('-'*80)
    print(glove_overlap)