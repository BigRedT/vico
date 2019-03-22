import h5py
import os
import numpy as np

import utils.io as io

exp_dir = os.path.join(
    os.getcwd(),
    'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms/dim_100_neg_bias_linear')
embed_dir = os.path.join(
        exp_dir,
        'concat_with_glove_300')
word_vecs_h5py = os.path.join(
    embed_dir,
    'visual_word_vecs.h5py')
word_to_idx_json = os.path.join(
    embed_dir,
    'visual_word_vecs_idx.json')

word_to_idx = io.load_json_object(word_to_idx_json)
embeddings = io.load_h5py_object(word_vecs_h5py)['embeddings']

def get_embeddings(words,embed_type='glove'):
    word_to_embed = {}
    for word in words:
        idx = word_to_idx[word]
        embed = embeddings[idx][()]
        if embed_type=='glove':
            word_to_embed[word] = embed[:300]
        elif embed_type=='visual':
            word_to_embed[word] = embed[300:]
        elif embed_type=='both':
            word_to_embed[word] = embed
        else:
            assert(False), 'embed_type not implemented'

    return word_to_embed

class FindAnalogies():
    def __init__(self,words,embed_type):
        self.words = words
        self.embed_type = embed_type
        self.word_to_embed = get_embeddings(self.words,self.embed_type)

    def pred_D_vec(self,A,B,C):
        C_vec = self.word_to_embed[C]
        B_vec = self.word_to_embed[B]
        A_vec = self.word_to_embed[A]
        # C_vec = self.normalize(self.word_to_embed[C])
        # B_vec = self.normalize(self.word_to_embed[B])
        # A_vec = self.normalize(self.word_to_embed[A])
        Dp_vec = (B_vec-A_vec)+C_vec
        return Dp_vec

    def cosine(self,x,y):
        x = self.normalize(x)
        y = self.normalize(y)
        return np.sum(x*y)

    def dot(self,x,y):
        return np.sum(x*y)

    def normalize(self,x):
        x = x / (np.linalg.norm(x,ord=2)+1e-6)
        return x

    def get_score(self,A,B,C,D):
        D_vec = self.word_to_embed[D]
        C_vec = self.word_to_embed[C]
        B_vec = self.word_to_embed[B]
        A_vec = self.word_to_embed[A]
        # D_vec = self.normalize(self.word_to_embed[D])
        # C_vec = self.normalize(self.word_to_embed[C])
        # B_vec = self.normalize(self.word_to_embed[B])
        # A_vec = self.normalize(self.word_to_embed[A])
        return self.cosine(D_vec,B_vec-A_vec+C_vec)
        #return self.cosine(D_vec-C_vec,B_vec-A_vec)

    def find(self,A,B,C,D_options):
        Dp_vec = self.pred_D_vec(A,B,C)
        best_D = None
        best_D_score = -1
        for word in D_options:
            if word in [A,B,C]: #[A,B,C]:
                continue
            score = self.get_score(A,B,C,word)#Dp_vec,self.word_to_embed[word])
            
            if score > best_D_score:
                best_D_score = score
                best_D = word
            #print(word,score)
        
        #print(self.get_analogy_str(A,B,C,best_D))
        return best_D, best_D_score

    def get_analogy_str(self,A,B,C,D=None):
        if D is None:
            return f'{A}:{B}::{C}:?'
        else:
            return f'{A}:{B}::{C}:{D}'


colors = ['red','blue','green','yellow','black','white'] #,'orange','brown','grey']
fruits = ['banana','strawberry','blueberry','apple','grapes','mango','orange','lemon']
people = ['man','male','woman','female','girl','boy','adult','child'] #'king','queen'
body = ['leg','arm','neck','chest','wrist']
animals = ['bear','zebra','dog','monkey','cat','sheep']
transport_medium = ['land','ocean','sky','road','railway']
stuff = ['grass','sky','ocean','sunlight','land','water','air','snow']
objects = ['leaf','sun','tv','clock','door','window','table','chair']
furniture = ['door','window','table','chair','sofa','sink','bench','cabinet','pew']
env = ['park','house','church','forest','zoo']
utensils = ['plate','glass','fork','knife','bowl']
clothes = ['hat','shirt','trouser','jeans','skirt','scarf','watch','shoe']
actions = ['sit','eat','drink','walk','talk','laugh','drive','fly','ride']
drinks = ['tea','coffee','water']
vehicles = ['car','bike','bicycle','cycle','bus','aeroplane','airplane','train','helicopter','boat','ship','scooter','air']
shapes = ['circle','square','triangle','octagon','round']
hypernyms = ['color','fruit','person','animal','action','drink','vehicle','shape','food']
material = ['wood','glass','ceramic','steel','concrete','paper','plastic']
words = fruits + colors + people + stuff + objects + clothes + actions + \
    drinks + vehicles + animals + hypernyms + shapes + body + material + \
    utensils + transport_medium + furniture + env + \
    ['hair','horn','coat','tail','fur'] + ['watch','shoe','tie','bandana'] + \
    ['spinach','rice'] + ['moustache','lipstick'] + ['king','queen'] + \
    ['can','bottle'] + ['metallic','cloth'] + ['market'] + ['desert']

finder = {}
for embed_type in ['glove','visual','both']:
    finder[embed_type] = FindAnalogies(words,embed_type)

analogy_tuples = [
    ['man','male','woman','female',people],
    ['sky','blue','leaf','green',colors],
    ['leaf','green','sky','blue',colors],
    ['apple','red','banana','yellow',colors],
    ['banana','yellow','apple','red',colors],
    ['yellow','banana','red','strawberry',['strawberry','lemon','mango','orange']],
    ['rice','white','spinach','green',['red','green','blue','yellow']],
    ['man','boy','woman','girl',people],
    ['car','drive','bicycle','ride',['eat','drink','talk','laugh','drive','fly','ride']],
    ['car','land','aeroplane','sky',transport_medium],
    ['car','land','ship','ocean',transport_medium],
    ['ship','ocean','train','railway',transport_medium],
    ['train','railway','car','road',['ocean','sky','road','railway','desert']],
    ['train','railway','car','land',['ocean','sky','land','railway','desert']],
    ['clock','circle','tv','square',shapes],
    ['man','moustache','woman','lipstick',['lipstick','watch','shirt','coat']],
    ['scarf','neck','watch','wrist',body],
    ['leg','trouser','wrist','watch',['watch','shoe','tie','bandana']],
    ['scarf','neck','trouser','leg',body],
    ['door','wood','window','glass',material],
    ['shoe','black','paper','white',colors],
    ['food','plate','water','glass',utensils],
    ['table','chair','door','window',furniture],
    ['park','bench','church','pew',furniture],
    ['monkey','zoo','dog','house',['house','church','forest','market']],
    ['monkey','zoo','cat','house',env],
    ['sheep','fur','person','hair',['hair','horn','coat','tail']],
    ['man','king','woman','queen',people+['king','queen']],
    ['can','metallic','bottle','plastic',['wood','plastic','cloth','paper']],
    ['can','metallic','bottle','glass',['wood','glass','cloth','paper']]
]

only_visual_correct = 0
only_glove_correct = 0
both_correct = 0 # different from embed_type both (when vico and glove are both correct)
both_wrong = 0
together_correct = 0
total = 0
for analogy_tuple in analogy_tuples:
    A,B,C,D,D_options = analogy_tuple
    print('-'*80)
    print(finder['glove'].get_analogy_str(A,B,C),
        [d for d in D_options if d not in [A,B,C]])
    print('-'*80)
    visual_correct_ = 0
    glove_correct_ = 0
    both_correct_ = 0
    both_wrong_ = 0
    together_correct_ = 0
    for embed_type in finder.keys():
        print_str = f'{embed_type} | '
        
        Dp,D_score = finder[embed_type].find(A,B,C,D_options)
        D_score = str(round(D_score,2))
        #print_str += f'{Dp} {D_score} (options) | '
        print_str += f'{Dp} {D_score} (pred)| '
            
        if embed_type=='glove' and Dp==D:
            glove_correct_ = 1

        if embed_type=='visual' and Dp==D:
            visual_correct_ = 1

        if embed_type=='both' and Dp==D:
            together_correct_ = 1
            together_correct += 1
        
        Dp,D_score = finder[embed_type].find(A,B,C,[D])
        D_score = str(round(D_score,2))
        print_str += f'{Dp} {D_score} (true) | '
        
        # Dp,D_score = finder[embed_type].find(A,B,C,words)
        # D_score = str(round(D_score,2))
        # print_str += f'{Dp} {D_score} (all)'
        
        print(print_str)
    
    total+=1
    if glove_correct_==1 and visual_correct_==1:
        both_correct += 1
    elif glove_correct_==0 and visual_correct_==0:
        both_wrong += 1
    elif glove_correct_==0 and visual_correct_==1:
        only_visual_correct += 1
    elif glove_correct_==1 and visual_correct_==0:
        only_glove_correct += 1
    else:
        assert(False), 'case not found'

print('Only GloVe',only_glove_correct)
print('Only Visual',only_visual_correct)
print('Both Correct',both_correct)
print('Both Incorrect',both_wrong)
print('Together Correct',together_correct)
print('Total',total)
#import pdb; pdb.set_trace()