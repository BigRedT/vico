import os
import copy
import h5py
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet as wn

import utils.io as io
from utils.constants import save_constants


def compute_norm(x):
    return np.linalg.norm(x,2,1)


def normalize(x):
    return x / (1e-6 + compute_norm(x)[:,np.newaxis])


class Reps():
    def __init__(self,const):
        self.const = copy.deepcopy(const)
        self.reps = None
        self.wnid_to_idx = None
        self.word_to_glove_idx = None

    @property
    def reps_dim(self):
        return self.reps.shape[-1]
    
    @property
    def num_glove_words(self):
        return len(self.word_to_glove_idx)

    @property
    def num_visual_wnids(self):
        return len(self.wnid_to_idx)

    @property
    def num_visual_words(self):
        return self.num_words_affected

    def init_word_reps(self):
        return np.zeros([self.num_glove_words,self.reps_dim],dtype=np.float32)

    def load_word_to_glove_idx(self,word_to_glove_idx_json):
        return io.load_json_object(word_to_glove_idx_json)

    def create_word_reps(self):
        word_reps = self.init_word_reps()
        freq = np.zeros([self.num_glove_words])
        words_affected = set()
        for wnid,idx in tqdm(self.wnid_to_idx.items()):
            rep = self.reps[idx]
            
            lemma_names = [lemma.name() for lemma in wn.synset(wnid).lemmas()]
            words = set()
            for lemma_name in lemma_names:
                lemma_words = lemma_name.lower().split('_')
                for word in lemma_words:
                    words.add(word)
            words = list(words)
            words_affected.update(words)

            for word in words:
                if word in self.word_to_glove_idx:
                    glove_idx = self.word_to_glove_idx[word]
                    word_reps[glove_idx] += rep
                    freq[glove_idx] += 1

        word_reps = word_reps / (freq[:,np.newaxis]+1e-6)
        word_reps = normalize(word_reps)
        num_words_affected = np.sum(freq > 0)

        # Fill in unaffected words with mean representation
        mean_rep = np.mean(self.reps,0)
        mean_rep = mean_rep / (np.linalg.norm(mean_rep,2,0)+1e-6)
        for word,idx in self.word_to_glove_idx.items():
            if word not in words_affected:
                word_reps[idx] = mean_rep

        return word_reps, freq, num_words_affected

    def words_affected(self):
        words = set()
        idx_to_word = {i:w for w,i in self.word_to_glove_idx.items()}
        for i in range(self.freq.shape[0]):
            if self.freq[i] > 0:
                words.add(idx_to_word[i])
        return words


class EntityEntityReps(Reps):
    def __init__(self,const):
        super(EntityEntityReps,self).__init__(const)
        self.const = copy.deepcopy(const)
        self.reps = self.load_reps(self.const.entity_entity_reps_npy)
        self.wnid_to_idx = self.load_wnid_to_idx(
            self.const.entity_wnid_offset_to_idx_json)
        self.word_to_glove_idx = self.load_word_to_glove_idx(
            self.const.word_to_glove_idx_json)
        self.word_reps, self.freq, self.num_words_affected = \
            self.create_word_reps()

    def load_reps(self,reps_npy):
        reps = np.load(reps_npy)
        return reps

    def load_wnid_to_idx(self,wnid_offset_to_idx_json):
        wnid_offset_to_idx = io.load_json_object(wnid_offset_to_idx_json)
        wnid_to_idx = {}
        for offset, idx in wnid_offset_to_idx.items():
            synset = wn.synset_from_pos_and_offset(offset[0],int(offset[1:]))
            wnid = synset.name()
            wnid_to_idx[wnid] = idx
        return wnid_to_idx


class EntityAttrReps(Reps):
    def __init__(self,const):
        super(EntityAttrReps,self).__init__(const)
        self.const = copy.deepcopy(const)
        self.reps = self.load_reps(self.const.entity_attr_reps_npy)
        self.wnid_to_idx = self.load_wnid_to_idx(
            self.const.entity_wnid_offset_to_idx_json)
        self.word_to_glove_idx = self.load_word_to_glove_idx(
            self.const.word_to_glove_idx_json)
        self.word_reps, self.freq, self.num_words_affected = \
            self.create_word_reps()

    def load_reps(self,reps_npy):
        reps = np.load(reps_npy)
        return reps

    def load_wnid_to_idx(self,wnid_offset_to_idx_json):
        wnid_offset_to_idx = io.load_json_object(wnid_offset_to_idx_json)
        wnid_to_idx = {}
        for offset, idx in wnid_offset_to_idx.items():
            synset = wn.synset_from_pos_and_offset(offset[0],int(offset[1:]))
            wnid = synset.name()
            wnid_to_idx[wnid] = idx
        return wnid_to_idx
    

class AttrAttrReps(Reps):
    def __init__(self,const):
        super(AttrAttrReps,self).__init__(const)
        self.const = copy.deepcopy(const)
        self.reps = self.load_reps(self.const.attr_attr_reps_npy)
        self.wnid_to_idx = self.load_wnid_to_idx(
            self.const.attr_wnid_to_idx_json)
        self.word_to_glove_idx = self.load_word_to_glove_idx(
            self.const.word_to_glove_idx_json)
        self.word_reps, self.freq, self.num_words_affected = \
            self.create_word_reps()

    def load_reps(self,reps_npy):
        reps = np.load(reps_npy)
        return reps

    def load_wnid_to_idx(self,wnid_to_idx_json):
        return io.load_json_object(wnid_to_idx_json)


class AttrEntityReps(Reps):
    def __init__(self,const):
        super(AttrEntityReps,self).__init__(const)
        self.const = copy.deepcopy(const)
        self.reps = self.load_reps(self.const.attr_entity_reps_npy)
        self.wnid_to_idx = self.load_wnid_to_idx(
            self.const.attr_wnid_to_idx_json)
        self.word_to_glove_idx = self.load_word_to_glove_idx(
            self.const.word_to_glove_idx_json)
        self.word_reps, self.freq, self.num_words_affected = \
            self.create_word_reps()

    def load_reps(self,reps_npy):
        reps = np.load(reps_npy)
        return reps

    def load_wnid_to_idx(self,wnid_to_idx_json):
        return io.load_json_object(wnid_to_idx_json)


class GloveReps(Reps):
    def __init__(self,const):
        super(GloveReps,self).__init__(const)
        self.const = copy.deepcopy(const)
        self.reps = self.load_reps(self.const.glove_h5py)
        self.word_to_glove_idx = self.load_word_to_glove_idx(
            self.const.word_to_glove_idx_json)
        self.word_reps, self.freq, self.num_words_affected = \
            self.create_word_reps()

    def load_reps(self,reps_h5py):
        return h5py.File(reps_h5py,'r')['embeddings'][()]

    def load_wnid_to_idx(self,wnid_to_idx_json):
        return io.load_json_object(wnid_to_idx_json)

    @property
    def num_visual_wnids(self):
        return 0

    @property
    def num_visual_words(self):
        return 0

    def create_word_reps(self):
        word_reps = self.reps
        freq = np.ones([self.num_glove_words])
        num_words_affected = self.num_glove_words
        return word_reps, freq, num_words_affected


def combine_reps(reps):
    # Compute entity reps
    entity_reps = reps['entity_entity'].word_reps*\
        (reps['entity_entity'].freq[:,np.newaxis]+1e-6)
    entity_freq = reps['entity_entity'].freq + 1e-6
    #import pdb; pdb.set_trace()
    if reps['attr_entity'] is not None:
        entity_reps += (reps['attr_entity'].word_reps*\
            (reps['attr_entity'].freq[:,np.newaxis]+1e-6))
        entity_freq += reps['attr_entity'].freq + 1e-6
    #import pdb; pdb.set_trace()
    entity_reps = entity_reps / (entity_freq[:,np.newaxis])
    #import pdb; pdb.set_trace()
    entity_reps = normalize(entity_reps)
    #import pdb; pdb.set_trace()

    # Compute attr reps
    attr_reps = reps['attr_attr'].word_reps*\
        (reps['attr_attr'].freq[:,np.newaxis]+1e-6)
    attr_freq = reps['attr_attr'].freq + 1e-6
    if reps['entity_attr'] is not None:
        attr_reps += (reps['entity_attr'].word_reps*\
            (reps['entity_attr'].freq[:,np.newaxis]+1e-6))
        attr_freq += reps['entity_attr'].freq + 1e-6
    attr_reps = attr_reps / (attr_freq[:,np.newaxis])
    attr_reps = normalize(attr_reps)

    # Concat with glove
    glove_reps = reps['glove'].word_reps
    combined_reps = np.concatenate((glove_reps,entity_reps,attr_reps),axis=1)
    return combined_reps


def main(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    save_constants(
        {'exp': exp_const, 'data': data_const},
        exp_const.exp_dir)

    print('Loading Entity-Entity Reps')
    entity_entity_reps = EntityEntityReps(data_const)
    print('Loading Entity-Attr Reps')
    entity_attr_reps = EntityAttrReps(data_const)
    print('Loading Attr-Attr Reps')
    attr_attr_reps = AttrAttrReps(data_const)
    print('Loading Attr-Entity Reps')
    attr_entity_reps = AttrEntityReps(data_const)
    print('Loading Glove Reps')
    glove_reps = GloveReps(data_const)

    print('Combining glove with visual reps ...')
    reps = {
        'glove': glove_reps,
        'entity_entity': entity_entity_reps,
        'entity_attr': entity_attr_reps,
        'attr_entity': attr_entity_reps,
        'attr_attr': attr_attr_reps,
    }
    combined_reps = combine_reps(reps)
    print('Saving combined reps to h5py file ...')
    visual_word_vecs_h5py =  h5py.File(
        os.path.join(exp_const.exp_dir,'visual_word_vecs.h5py'),
        'w')
    visual_word_vecs_h5py.create_dataset(
        'embeddings',
        data=combined_reps,
        chunks=(1,combined_reps.shape[-1]))
    visual_word_vecs_h5py.close()

    visual_word_vecs_idx_json = os.path.join(
        exp_const.exp_dir,
        'visual_word_vecs_idx.json')
    io.dump_json_object(
        glove_reps.word_to_glove_idx,
        visual_word_vecs_idx_json)

    visual_words = set()
    for rep_type in reps.keys():
        if rep_type=='glove':
            continue
        visual_words.update(reps[rep_type].words_affected())
    visual_words_json = os.path.join(exp_const.exp_dir,'visual_words.json')
    io.dump_json_object(list(visual_words),visual_words_json)
    print(f'Number of visual words: {len(visual_words)}')

    freq = None
    for rep_type in reps.keys():
        if rep_type=='glove':
            continue
        
        if freq is None:
            freq = reps[rep_type].freq
        else:
            freq += reps[rep_type].freq

    visual_freq = {}
    for word,idx in glove_reps.word_to_glove_idx.items():
        if freq[idx] > 0:
            visual_freq[word] = freq[idx]

    visual_freq_json = os.path.join(exp_const.exp_dir,'visual_freq.json')
    io.dump_json_object(visual_freq,visual_freq_json)