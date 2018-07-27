import os

import utils.io as io
from tqdm import tqdm
from data.semeval_2018_10.constants import SemEval201810Constants

def compute_word_freq(words):
    freqs = {}
    for row in words:
        for word in row:
            if word not in freqs:
                freqs[word] = 1
            else:
                freqs[word] += 1
    return freqs


def main():
    const = SemEval201810Constants()

    subset_json = {
        'train': const.train_json,
        'val': const.val_json,
        'test': const.test_json
    }

    words = []
    for subset, json_file in subset_json.items():
        data = io.load_json_object(json_file)
        data = [row[:2] for row in data]
        words += data

    print('Computing word frequency ...')
    word_freqs = compute_word_freq(words)
    print(f'Number of words: {len(word_freqs)}')
    io.dump_json_object(
        word_freqs,
        os.path.join(const.proc_dir,'word_freqs.json'))


if __name__=='__main__':
    main()