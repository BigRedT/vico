import os

import utils.io as io


class SemEval201810Constants(io.JsonSerializableClass):
    def __init__(
            self,
            raw_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/semeval_2018_10/raw'),
            proc_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/semeval_2018_10/proc')):
        self.raw_dir = raw_dir
        self.proc_dir = proc_dir

        self.train_txt = os.path.join(
            self.raw_dir,
            'training/train.txt')
        self.val_txt = os.path.join(
            self.raw_dir,
            'training/validation.txt')
        self.test_txt = os.path.join(
            self.raw_dir,
            'test/test_triples.txt')
        self.truth_txt = os.path.join(
            self.raw_dir,
            'test/ref/truth.txt')

        self.train_json = os.path.join(
            self.proc_dir,
            'train.json')
        self.val_json = os.path.join(
            self.proc_dir,
            'val.json')
        self.test_json = os.path.join(
            self.proc_dir,
            'test.json')
        self.truth_json = os.path.join(
            self.proc_dir,
            'truth.json')
        self.word_freqs = os.path.join(
            self.proc_dir,
            'word_freqs.json')
        self.feature_freqs = os.path.join(
            self.proc_dir,
            'feature_freqs.json')
        self.all_word_freqs = os.path.join(
            self.proc_dir,
            'all_words.json')
