import os
import re
import copy
import itertools
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

import utils.io as io
from utils.constants import save_constants
from utils.lemmatizer import Lemmatizer


def create_object_list(object_annos,obj_ids):
    objects = set()
    for obj_id in obj_ids:
        anno = object_annos[obj_id]
        region_objects = set(anno['names'])
        objects.update(region_objects)

    return list(objects)


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


def main(exp_const,data_const):
    print(f'Creating directory {exp_const.exp_dir} ...')
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)

    print('Saving constants ...')
    save_constants(
        {'exp': exp_const,'data': data_const},
        exp_const.exp_dir)

    print('Loading data ...')
    img_id_to_obj_id = io.load_json_object(
        data_const.image_id_to_object_id_json)
    object_annos = io.load_json_object(data_const.object_annos_json)

    cooccur = {}
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    lemmatizer = Lemmatizer()
    for img_id, obj_ids in tqdm(img_id_to_obj_id.items()):
        object_list = create_object_list(object_annos,obj_ids)
        object_words_set = get_words_from_phrases(
            object_list,
            lemmatizer,
            stop_words)
        for word1 in object_words_set:
            for word2 in object_words_set:
                if word1 not in cooccur:
                        cooccur[word1] = {}
                    
                if word2 not in cooccur[word1]:
                    cooccur[word1][word2] = 0

                cooccur[word1][word2] += 1

    word_cooccur_json = os.path.join(exp_const.exp_dir,'word_cooccur.json')
    io.dump_json_object(cooccur,word_cooccur_json)
                


