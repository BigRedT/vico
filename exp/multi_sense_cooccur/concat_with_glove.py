import os
import h5py
from tqdm import tqdm
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

import utils.io as io
from utils.constants import save_constants


class Lemmatizer():
    def __init__(self):
        self.lemmatize_ = WordNetLemmatizer().lemmatize
        self.pos_tag_ = nltk.pos_tag

    def lemmatize(self,word):
        pos_tag = self.get_pos(word)
        if pos_tag is None:
            return word
        
        return self.lemmatize_(word,pos=pos_tag)

    def get_pos(self,word):
        pos_tag = self.pos_tag_([word])[0][1]
        if pos_tag in ['NN','NNS','NNP','NNPS']:
            return 'n'
        elif pos_tag in ['JJ','JJR','JJS']:
            return 'a'
        elif pos_tag in ['VB','VBZ','VBG','VBD','VBN','VBP']:
            return 'v'
        else:
            return None


def compute_norm(x):
    return np.linalg.norm(x,2)


def normalize(x):
    return x / (1e-6 + compute_norm(x))


def main(exp_const,data_const):
    nltk.download('averaged_perceptron_tagger')
    
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    save_constants(
        {'exp': exp_const, 'data': data_const},
        exp_const.exp_dir)
    
    print('Loading glove embeddings ...')
    glove_idx = io.load_json_object(data_const.glove_idx)
    glove_h5py = h5py.File(
        data_const.glove_h5py,
        'r')
    glove_embeddings = glove_h5py['embeddings'][()]
    num_glove_words, glove_dim = glove_embeddings.shape
    print('-'*80)
    print(f'number of glove words: {num_glove_words}')
    print(f'glove dim: {glove_dim}')
    print('-'*80)

    print('Loading visual embeddings ...')
    visual_embeddings = np.load(data_const.visual_embeddings_npy)
    visual_word_to_idx = io.load_json_object(data_const.visual_word_to_idx)
    # io.dump_json_object(
    #     list(visual_word_to_idx.keys()),
    #     os.path.join(exp_const.exp_dir,'visual_words.json'))
    mean_visual_embedding = np.mean(visual_embeddings,0)
    num_visual_words, visual_embed_dim = visual_embeddings.shape
    print('-'*80)
    print(f'number of visual embeddings: {num_visual_words}')
    print(f'visual embedding dim: {visual_embed_dim}')
    print('-'*80)

    print('Combining glove with visual embeddings ...')
    visual_word_vecs_idx_json = os.path.join(
        exp_const.exp_dir,
        'visual_word_vecs_idx.json')
    io.dump_json_object(glove_idx,visual_word_vecs_idx_json)
    visual_word_vecs_h5py =  h5py.File(
        os.path.join(exp_const.exp_dir,'visual_word_vecs.h5py'),
        'w') 
    visual_word_vec_dim = glove_dim + visual_embed_dim
    visual_word_vecs = np.zeros([num_glove_words,visual_word_vec_dim])
    visual_words = set()
    lemmatizer = Lemmatizer()
    for word in tqdm(glove_idx.keys()):
        glove_id = glove_idx[word]
        glove_vec = glove_embeddings[glove_id]
        
        if word in visual_word_to_idx:
            idx = visual_word_to_idx[word]
            visual_embedding = visual_embeddings[idx]
            visual_words.add(word)
        else:
            lemma = lemmatizer.lemmatize(word)
            if lemma in visual_word_to_idx:
                idx = visual_word_to_idx[lemma]
                visual_embedding = visual_embeddings[idx]
                visual_words.add(lemma)
                visual_words.add(word)
            else:
                visual_embedding = mean_visual_embedding
        
        visual_word_vec = np.concatenate((
            glove_vec,
            visual_embedding))
        visual_word_vecs[glove_id] = visual_word_vec
    
    visual_word_vecs_h5py.create_dataset(
        'embeddings',
        data=visual_word_vecs,
        chunks=(1,visual_word_vec_dim))
    visual_word_vecs_h5py.close()

    io.dump_json_object(
        list(visual_words),
        os.path.join(exp_const.exp_dir,'visual_words.json'))