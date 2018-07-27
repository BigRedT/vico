import os

import utils.io as io
from tqdm import tqdm
from data.semeval_2018_10.constants import SemEval201810Constants

def compute_feature_freq(features):
    freqs = {}
    for word in features:
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

    features = []
    for subset, json_file in subset_json.items():
        data = io.load_json_object(json_file)
        data = [row[2] for row in data]
        features += data

    print('Computing feature frequency ...')
    feature_freqs = compute_feature_freq(features)
    print(f'Number of features: {len(feature_freqs)}')
    io.dump_json_object(
        feature_freqs,
        os.path.join(const.proc_dir,'feature_freqs.json'))


if __name__=='__main__':
    main()