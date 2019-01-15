import os
import re
import nltk 
from tqdm import tqdm
from nltk.corpus import wordnet

from .constants import VisualGenomeConstants
import utils.io as io

def attr_group_test(synset,def_words,test_words):
    for word in test_words:
        if word == synset.name().split('.')[0]:
            return True

        found = findWholeWord(word,def_words)
        if found is not None:
            return True
    
    return False


# def attr_group_test(synset,test_synsets):
#     for ancestor in synset.hypernyms():
#         if ancestor.name() in test_synsets:
#             return True
    
#     return False


def findWholeWord(word,sentence):
    pattern = re.compile(r'\b({0})\b'.format(word), flags=re.IGNORECASE)
    return pattern.search(sentence)


def get_synsets(synset_strs):
    return [wordnet.synset(s) for s in synset_strs]


def main():
    const = VisualGenomeConstants()
    attribute_synset_freqs = io.load_json_object(const.attribute_synset_freqs_json)
    attr_groups = {
        'shape': set(),
        'color': set(),
        'material': set(),
        'noun': set(),
        'adj': set(),
        'adj_sat': set(),
        'adv': set(),
        'verb': set()
    }
    
    # shape_synsets = ['form.n.07','form.n.03','shape.n.01','shape.n.02','shape.n.04']
    # color_synsets = ['color.n.01','color.n.08','color.a.01','coloring_material.n.01']
    # material_synsets = ['material.n.01','fabric.n.01','material.a.02']
    # texture_synsets = ['texture.n.01','texture.n.02','texture.n.05']

    color_words = ['color','shade','chroma','chromatic','light','bright','dark']
    shape_words = ['shape','shaped','edges','edge','edged','sharp',
        'sharpened','line','angle','angles','angular','rectangular','rectangle',
        'circular','elliptical','circle','elipse','tall','short','thin',
        'wide']
    material_words = ['material','materials','surface','surfaces','rough',
        'smooth','made of','made or consisting of','solid','liquid','gas',
        'through','transparent','glass','glassy','shine','shiny','reflecting',
        'metal','metallic','hot','cold','frozen','cooked','uncooked','leaf',
        'leafy','dust','dusty','texture','pattern','patterned','grainy','feel']

    for synset_str in tqdm(attribute_synset_freqs.keys()):
        w = wordnet.synset(synset_str)
        pos = synset_str.split('.')[-2]
        def_words = w.definition().lower()#.split(' ')
        color_test = attr_group_test(w,def_words,color_words)
        shape_test = attr_group_test(w,def_words,shape_words)
        material_test = attr_group_test(w,def_words,material_words)
        if pos == 'r':
            group = 'adv'
        elif pos == 'v':
            group = 'verb'
        elif pos =='n':
            group = 'noun'
        elif color_test==True:
            group = 'color'
        elif shape_test==True:
            group = 'shape'
        elif material_test==True:
            group = 'material'
        elif pos == 'a':
            group = 'adj'
        elif pos == 's':
            group = 'adj_sat'
        else:
            import pdb; pdb.set_trace()
            assert(False), 'Group Not found'
        
        attr_groups[group].add(synset_str)

    for k in attr_groups.keys():
        attr_groups[k] = list(attr_groups[k])
        
    attribute_groups_json = os.path.join(const.proc_dir,'attribute_groups.json')
    io.dump_json_object(attr_groups,attribute_groups_json)


if __name__=='__main__':
    main()