import os
import itertools

import utils.io as io
from data.semeval_2018_10.constants import SemEval201810Constants
from data.visualgenome.constants import VisualGenomeConstants


def main():
    semeval_const = SemEval201810Constants()
    vgenome_const = VisualGenomeConstants()

    object_freqs = io.load_json_object(vgenome_const.object_freqs_json)
    attribute_freqs = io.load_json_object(vgenome_const.attribute_freqs_json)

    word_freqs = io.load_json_object(semeval_const.word_freqs)
    feature_freqs = io.load_json_object(semeval_const.feature_freqs)    

    word_overlap = 0
    for word in word_freqs.keys():
        if (word in object_freqs) or (word in attribute_freqs):
            word_overlap += 1

    frac_word_overlap = round(100 * word_overlap / len(word_freqs),2)
    print(f'Word overlap: {frac_word_overlap}')

    feature_overlap = 0
    for feature in feature_freqs.keys():
        if (feature in object_freqs) or (feature in attribute_freqs):
            feature_overlap += 1

    frac_feature_overlap = round(100 * feature_overlap / len(feature_freqs),2)
    print(f'Feature overlap: {frac_feature_overlap}')

if __name__=='__main__':
    main()