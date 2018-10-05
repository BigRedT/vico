import copy

import utils.io as io
from data.visualgenome.constants import VisualGenomeConstants

def main():
    genome_const = VisualGenomeConstants()
    object_freqs = io.load_json_object(genome_const.object_freqs_json)
    attribute_freqs = io.load_json_object(genome_const.attribute_freqs_json)
    vocab = copy.deepcopy(object_freqs)
    for word,freq in attribute_freqs.items():
        if word not in vocab:
            vocab[word] = 0

        vocab[word] += freq

    io.dump_json_object(vocab, genome_const.all_word_freqs_json)


if __name__=='__main__':
    main()