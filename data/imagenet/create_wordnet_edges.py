import os
from tqdm import tqdm

import utils.io as io
from .constants import ImagenetConstants

def main():
    const = ImagenetConstants()
    
    print('Reading txt file ...')
    with open(const.is_a_txt,'r',encoding='ISO-8859-1') as f: #ISO-8859-1
        lines = f.readlines()

    print('Parsing is_a relationship ...')
    lines = [line.rstrip('\n') for line in lines]
    lines = [line.split(' ') for line in lines]
    wnid_to_parent = {c:p for p,c in lines}
    
    print('Saving wnid_to_parent.json')
    io.dump_json_object(wnid_to_parent,const.wnid_to_parent_json)


if __name__ == "__main__":
    main()