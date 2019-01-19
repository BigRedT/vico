import os
import copy
import itertools
from tqdm import tqdm
from nltk.corpus import wordnet as wn

import utils.io as io
from utils.constants import save_constants


def create_synset_list(object_annos,obj_ids):
    synsets = set()
    for obj_id in obj_ids:
        anno = object_annos[obj_id]
        region_synsets = set(anno['object_synsets'])
        synsets.update(region_synsets)

    return list(synsets)


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
    for img_id, obj_ids in tqdm(img_id_to_obj_id.items()):
        synset_list = create_synset_list(object_annos,obj_ids)
        for synset1 in synset_list:
            for synset2 in synset_list:
                if synset1 not in cooccur:
                        cooccur[synset1] = {}
                    
                if synset2 not in cooccur[synset1]:
                    cooccur[synset1][synset2] = 0

                cooccur[synset1][synset2] += 1

    synset_cooccur_json = os.path.join(exp_const.exp_dir,'synset_cooccur.json')
    io.dump_json_object(cooccur,synset_cooccur_json)
                


