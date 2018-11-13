import os
from tqdm import tqdm

import utils.io as io
from .constants import ImagenetConstants

def main():
    const = ImagenetConstants()
    words_txt = const.words_txt
    
    print('Reading txt file ...')
    with open(words_txt,'r') as f: #ISO-8859-1
        lines = f.readlines()

    print('Parse each row to get wnid and words ...')
    lines = [line.rstrip('\n').split('\t') for line in lines]
    lines = {k:v.split(', ') for k,v in lines}

    wnid_to_urls_json = const.wnid_to_urls_json
    wnids = io.load_json_object(wnid_to_urls_json).keys()

    wnid_to_words = {}
    for wnid in wnids:
        wnid_to_words[wnid] = lines[wnid]
    
    wnid_to_words_json = const.wnid_to_words_json
    io.dump_json_object(wnid_to_words,wnid_to_words_json)

    num_wnids = len(wnid_to_words)
    print(f'Number of wnids: {num_wnids}') 

if __name__=='__main__':
    main()