import os
from tqdm import tqdm

import utils.io as io
from .constants import ImagenetConstants

def main():
    const = ImagenetConstants()
    urls_txt = const.urls_txt
    
    print('Reading txt file ...')
    with open(urls_txt,'r',encoding='ISO-8859-1') as f: #ISO-8859-1
        lines = f.readlines()
        
    print('Parsing each row to get wnid, imgid, and url ...')
    wnid_to_urls = {}
    lines_skipped = 0
    for line in tqdm(lines):
        try:
            line = line.rstrip('\n')
            wnid_subwnid, url = line.split('\t')
            wnid, subwnid = wnid_subwnid.split('_')
        except:
            lines_skipped += 1
            continue
        
        if wnid not in wnid_to_urls:
            wnid_to_urls[wnid] = {}
        
        wnid_to_urls[wnid][subwnid] = url

    wnid_count = len(wnid_to_urls)
    img_count = 0
    for wnid, subwnid_to_url in wnid_to_urls.items():
        img_count += len(subwnid_to_url)
    
    print(f'Image Count: {img_count}')
    print(f'Wordnet ID Count: {wnid_count}')
    print(f'Images skipped: {lines_skipped}')

    wnid_to_urls_json = const.wnid_to_urls_json
    import pdb; pdb.set_trace()
    io.dump_json_object(wnid_to_urls,wnid_to_urls_json)
    

if __name__=='__main__':
    main()
