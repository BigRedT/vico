import os
import re
import copy
import itertools
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from torch.utils.data import DataLoader

import utils.io as io
from utils.constants import save_constants
from .dataset import GenomeAttributesNoImgsDataset
from utils.lemmatizer import Lemmatizer


def get_words_from_phrases(phrases,lemmatizer,stop_words):
    words = set()
    for phrase in phrases:
        phrase = phrase.lower()
        phrase = re.sub('[-]',' ',phrase)
        phrase = re.sub('[^a-zA-Z ]','',phrase)
        for word in phrase.lower().split(' '):
            if word=='' or word in stop_words:
                continue
            
            lemma = lemmatizer.lemmatize(word)
            words.add(lemma)

    return words


def create_gt_word_cooccur(exp_const,dataloader):
    print('Creating cooccur ...')
    cooccur = {}
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    lemmatizer = Lemmatizer()
    for data in tqdm(dataloader):
        B = len(data['object_words'])

        for b in range(B):
            object_words_set = get_words_from_phrases(
                data['object_words'][b],
                lemmatizer,
                stop_words)
            attribute_words_set = get_words_from_phrases(
                data['attribute_words'][b],
                lemmatizer,
                stop_words)
            for word1 in object_words_set:
                for word2 in attribute_words_set:
                    if word1 not in cooccur:
                        cooccur[word1] = {}
                    
                    if word2 not in cooccur[word1]:
                        cooccur[word1][word2] = 0

                    cooccur[word1][word2] += 1

                    if word2 not in cooccur:
                        cooccur[word2] = {}
                    
                    if word1 not in cooccur[word2]:
                        cooccur[word2][word1] = 0

                    cooccur[word2][word1] += 1

            for word1 in object_words_set:
                if word1 not in cooccur[word1]:
                    cooccur[word1][word1] = 0
                
                cooccur[word1][word1] += 1

            for word2 in attribute_words_set:
                if word2 not in cooccur[word2]:
                    cooccur[word2][word2] = 0
                
                cooccur[word2][word2] += 1
        
    word_cooccur_json = os.path.join(exp_const.exp_dir,'word_cooccur.json')
    io.dump_json_object(cooccur,word_cooccur_json)

    print('Checking symmetry and self constraint in word cooccur ...')
    for word1, context in tqdm(cooccur.items()):
        for word2, count in context.items():
            sym_err_msg = f'Word cooccurence not symmetric ({word1} / {word2})'
            assert(cooccur[word2][word1]==count), err_msg

    print('Constraints satisfied')
    

def main(exp_const,data_const):
    print(f'Creating directory {exp_const.exp_dir} ...')
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)

    print('Saving constants ...')
    save_constants(
        {'exp': exp_const,'data': data_const},
        exp_const.exp_dir)

    print('Creating dataloader ...')
    data_const = copy.deepcopy(data_const)
    dataset = GenomeAttributesNoImgsDataset(data_const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets',
        'object_words','attribute_words',
        'attribute_labels_idxs'])
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=False,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    create_gt_word_cooccur(exp_const,dataloader)
    