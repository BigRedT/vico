import copy

import utils.io as io
from data.semeval_2018_10.constants import SemEval201810Constants

def main():
    semeval_const = SemEval201810Constants()
    word_freqs = io.load_json_object(semeval_const.word_freqs)
    feature_freqs = io.load_json_object(semeval_const.feature_freqs)
    vocab = copy.deepcopy(word_freqs)
    for word,freq in feature_freqs.items():
        if word not in vocab:
            vocab[word] = 0

        vocab[word] += freq

    io.dump_json_object(vocab, semeval_const.all_word_freqs)


if __name__=='__main__':
    main()