import os
import itertools
from tqdm import tqdm
from nltk.corpus import wordnet as wn

import utils.io as io


def synset_to_words(synset):
    lemma_names = [lemma.name() for lemma in wn.synset(synset).lemmas()]
    words = set()
    for lemma_name in lemma_names:
        lemma_words = lemma_name.lower().split('_')
        for word in lemma_words:
            if word.endswith("'s"):
                word = word[:-2]
            words.add(word)
    words = list(words)
    return words


def main(exp_const,data_const):
    print('Loading synset cooccurence ...')
    synset_cooccur = io.load_json_object(data_const.synset_cooccur_json)

    print('Checking symmetry and self constraint in synset cooccur ...')
    sym_err_msg = 'Word cooccurence not symmetric ...'
    self_err_msg = 'Self constraints violated ...'
    for word1, context in tqdm(synset_cooccur.items()):
        for word2, count in context.items():
            assert(synset_cooccur[word2][word1]==count), err_msg
            assert(synset_cooccur[word1][word1]>=count), self_err_msg
            assert(synset_cooccur[word2][word2]>=count), self_err_msg

    print('Mapping synsets to words ...')
    synset_to_words_dict = {}
    for synset in tqdm(synset_cooccur.keys()):
        synset_to_words_dict[synset] = synset_to_words(synset)

    print('Creating word cooccurrence ...')
    word_cooccur = {}
    for wnid1, context in tqdm(synset_cooccur.items()):
        words1 = synset_to_words_dict[wnid1]

        for wnid2, count in context.items():
            words2 = synset_to_words_dict[wnid2]
            
            # if wnid2!=wnid1:
            #     words2_list = list(set(words2)-set(words1))
            #     words1_list = list(set(words1)-set(words2))
            # else:
            #     words2_list = list(set(words2))
            #     words1_list = list(set(words1))

            # words1_list = list(set(words1))
            # words2_list = list(set(words2))
            words_list = list(set(words1+words2))
            for word1,word2 in itertools.product(words_list,words_list):
                if word1 not in word_cooccur:
                    word_cooccur[word1] = {}

                if word2 not in word_cooccur[word1]:
                    word_cooccur[word1][word2] = 0
                word_cooccur[word1][word2] += count

                # if word1 not in words2:
                #     if word1 not in word_cooccur[word1]:
                #         word_cooccur[word1][word1] = 0    
                #     word_cooccur[word1][word1] += count

    io.dump_json_object(word_cooccur,data_const.word_cooccur_json)

    print('Checking symmetry and self constraint in word cooccur...')
    for word1, context in tqdm(word_cooccur.items()):
        for word2, count in context.items():
            sym_err_msg = f'Word cooccurence not symmetric ({word1} / {word2})'
            self_err_msg = f'Self constraints violated ({word1} / {word2})'
            assert(word_cooccur[word2][word1]==count), err_msg
            assert(word_cooccur[word1][word1]>=count), self_err_msg
            assert(word_cooccur[word2][word2]>=count), self_err_msg

    print('Constraints satisfied')