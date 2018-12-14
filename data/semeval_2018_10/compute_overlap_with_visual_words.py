import os

import utils.io as io
from .constants import SemEval201810Constants


def compute_overlap(semeval_vocab,visual_vocab):
    intersection = semeval_vocab.intersection(visual_vocab)
    frac = len(intersection) / (len(semeval_vocab) +  1e-6)
    frac = round(frac*100,2)
    return frac


if __name__=='__main__':
    semeval_const = SemEval201810Constants()
    word_freqs = io.load_json_object(semeval_const.word_freqs)
    feature_freqs = io.load_json_object(semeval_const.feature_freqs)
    semeval_vocab = {
        'word': set(word_freqs.keys()),
        'feature': set(feature_freqs.keys()),
    }
    
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
    for k,v in semeval_vocab.items():
        visual_overlap[k] =  compute_overlap(v,set(visual_vocab))
        glove_overlap[k] = compute_overlap(v,set(glove_vocab.keys()))
    
    print('-'*80)
    print('Overlap with visual vocab')
    print('-'*80)
    print(visual_overlap)

    print('-'*80)
    print('Overlap with glove vocab')
    print('-'*80)
    print(glove_overlap)