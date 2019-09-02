import os
import copy
import itertools
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from torch.utils.data import DataLoader

import utils.io as io


def synset_to_words(synset,stop_words):
    lemma_names = [lemma.name() for lemma in synset.lemmas()]
    words = set()
    for lemma_name in lemma_names:
        lemma_name = lemma_name.lower()    
        lemma_words = lemma_name.split('_')
        
        for word in lemma_words:
            if word.endswith("'s"):
                word = word[:-2]
            words.add(word)
    
    words = list(words)
    return words


def main(exp_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)

    cooccur = {}
    nltk.download('wordnet')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    for synset in wn.all_synsets():
        words = synset_to_words(synset,stop_words)
        for word1 in words:
            for word2 in words:
                if word1 not in cooccur:
                        cooccur[word1] = {}

                if word2 not in cooccur[word1]:
                    cooccur[word1][word2] = 0

                cooccur[word1][word2] += 1

    cooccur_json = os.path.join(exp_const.exp_dir,'word_cooccur.json')
    io.dump_json_object(cooccur,cooccur_json)