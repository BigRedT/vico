import os
import h5py
from tqdm import tqdm 
import numpy as np

import utils.io as io
from utils.constants import save_constants


def aggregate_synset_features(
        synset_to_object_ids,
        synset_to_features_idx,
        object_id_to_features_idx,
        visual_features_h5py,
        synset_features_h5py):
    feat_dim = visual_features_h5py['features'].shape[1]
    visual_features = visual_features_h5py['features'][()]
    num_synsets = len(synset_to_features_idx)
    print('Initializing synset features h5py ...')
    synset_features_h5py.create_dataset(
        'features',
        (num_synsets,feat_dim))
    
    print('Iterating over synsets ...')
    for synset, object_ids in tqdm(synset_to_object_ids.items()):
        object_ids = [o for o in object_ids if o in object_id_to_features_idx]
        num_regions = len(object_ids)
        if num_regions==0:
            continue
    
        idxs = [object_id_to_features_idx[object_id] for object_id in object_ids]
        agg_features = np.mean(visual_features[idxs],0)

        idx = synset_to_features_idx[synset]
        synset_features_h5py['features'][idx] = agg_features


def merge_object_attribute_synset_features(
        object_synset_to_idx,
        attribute_synset_to_idx,
        object_visual_features_h5py,
        attribute_visual_features_h5py,
        word_to_idx,
        word_visual_features_h5py):
    word_freq = {}
    word_features = []
    word_count = 0
    for i, synset in enumerate(tqdm(object_synset_to_idx.keys())):
        word = synset.split('.')[0]
        
        if word not in word_to_idx:
            word_to_idx[word] = word_count
            word_count += 1

        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1
        
        object_feat_idx = object_synset_to_idx[synset]
        object_feat = object_visual_features_h5py['features'][object_feat_idx]
        
        word_feat_idx = word_to_idx[word]
        if word_feat_idx < len(word_features):
            word_features[word_feat_idx] = \
                word_features[word_feat_idx] + \
                object_feat
        elif word_feat_idx==len(word_features):
            word_features.append(object_feat)
        else:
            assert_str = 'word_feat_idx can only be <= len(word_features)'
            assert(False), assert_str
        
    for i, synset in enumerate(tqdm(attribute_synset_to_idx.keys())):
        word = synset.split('.')[0]
        
        if word not in word_to_idx:
            word_to_idx[word] = word_count
            word_count += 1

        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1
        
        attribute_feat_idx = attribute_synset_to_idx[synset]
        attribute_feat = \
            attribute_visual_features_h5py['features'][attribute_feat_idx]
        
        word_feat_idx = word_to_idx[word]
        if word_feat_idx < len(word_features):
            word_features[word_feat_idx] = \
                word_features[word_feat_idx] + \
                attribute_feat
        elif word_feat_idx==len(word_features):
            word_features.append(attribute_feat)
        else:
            assert_str = 'word_feat_idx can only be <= len(word_features)'
            assert(False), assert_str

    for word, idx in word_to_idx.items():
        freq = word_freq[word]
        word_features[idx] = word_features[idx] / freq

    word_features = np.stack(word_features)
    mean_word_feature = np.mean(word_features,0)
    word_visual_features_h5py.create_dataset(
        'features',
        data=word_features)
    word_visual_features_h5py.create_dataset(
        'mean',
        data=mean_word_feature)


def main(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir)
    save_constants(
        {'exp': exp_const, 'data': data_const},
        exp_const.exp_dir)

    print('Loading Visual Genome object and attribute synsets ...')
    object_synsets_freqs = io.load_json_object(
        data_const.object_synset_freqs_json)
    attribute_synset_freqs = io.load_json_object(
        data_const.attribute_synset_freqs_json)
    object_synsets = list(object_synsets_freqs.keys())
    attribute_synsets = list(attribute_synset_freqs.keys())
    object_synset_to_idx = {k:v for v,k in enumerate(object_synsets)}
    attribute_synset_to_idx = {k:v for v,k in enumerate(attribute_synsets)}

    print('Loading object_annos.json ...')
    object_annos = io.load_json_object(data_const.object_annos_json)
    
    print('Creating map from object synsets to object ids ...')
    object_synset_to_object_ids = {}
    for object_id, anno in tqdm(object_annos.items()):
        for object_synset in anno['object_synsets']:
            if object_synset not in object_synset_to_object_ids:
                object_synset_to_object_ids[object_synset] = []
            object_synset_to_object_ids[object_synset].append(object_id)
    
    object_synset_to_object_ids_json = os.path.join(
        exp_const.exp_dir,
        'object_synset_to_object_ids.json')
    io.dump_json_object(
        object_synset_to_object_ids,
        object_synset_to_object_ids_json)

    print('Creating map from attribute synsets to object ids ...')
    attribute_synset_to_object_ids = {}
    for object_id, anno in tqdm(object_annos.items()):
        for attribute_synset in anno['attribute_synsets']:
            if attribute_synset not in attribute_synset_to_object_ids:
                attribute_synset_to_object_ids[attribute_synset] = []
            attribute_synset_to_object_ids[attribute_synset].append(object_id)
    
    attribute_synset_to_object_ids_json = os.path.join(
        exp_const.exp_dir,
        'attribute_synset_to_object_ids.json')
    io.dump_json_object(
        attribute_synset_to_object_ids,
        attribute_synset_to_object_ids_json)

    print('Loading visual features ...')
    visual_features_h5py = h5py.File(data_const.visual_features_h5py,'r')
    object_id_to_features_idx = io.load_json_object(
        data_const.object_id_to_features_idx_json)

    print('Aggregating visual features for object synsets ...')
    object_visual_features_h5py = h5py.File(
        os.path.join(exp_const.exp_dir,'object_features.h5py'),
        'w')
    aggregate_synset_features(
        object_synset_to_object_ids,
        object_synset_to_idx,
        object_id_to_features_idx,
        visual_features_h5py,
        object_visual_features_h5py)
    object_visual_features_h5py.close()

    print('Aggregating visual features for attribute synsets ...')
    attribute_visual_features_h5py = h5py.File(
        os.path.join(exp_const.exp_dir,'attribute_features.h5py'),
        'w')
    aggregate_synset_features(
        attribute_synset_to_object_ids,
        attribute_synset_to_idx,
        object_id_to_features_idx,
        visual_features_h5py,
        attribute_visual_features_h5py)
    attribute_visual_features_h5py.close()

    print('Merge object and attribute synset features to get word features ...')
    object_visual_features_h5py = h5py.File(
        os.path.join(exp_const.exp_dir,'object_features.h5py'),
        'r')        
    attribute_visual_features_h5py = h5py.File(
        os.path.join(exp_const.exp_dir,'attribute_features.h5py'),
        'r')
    
    word_to_idx = {}
    word_visual_features_h5py = h5py.File(
        os.path.join(exp_const.exp_dir,'word_features.h5py'),
        'w')
    merge_object_attribute_synset_features(
        object_synset_to_idx,
        attribute_synset_to_idx,
        object_visual_features_h5py,
        attribute_visual_features_h5py,
        word_to_idx,
        word_visual_features_h5py)
    word_visual_features_h5py.close()   
    word_to_idx_json = os.path.join(
        exp_const.exp_dir,
        'word_to_idx.json')
    io.dump_json_object(word_to_idx,word_to_idx_json)