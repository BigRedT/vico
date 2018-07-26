import os

import utils.io as io
from tqdm import tqdm
from data.visualgenome.constants import VisualGenomeConstants


def compute_attribute_freqs(object_annos):
    freqs = {}
    for anno in tqdm(object_annos.values()):
        for synset in anno['attribute_synsets']:
            attribute = synset.split('.')[0]
            if attribute not in freqs:
                freqs[attribute] = 1
            else:
                freqs[attribute] += 1
    return freqs


def compute_attribute_synset_freqs(object_annos):
    freqs = {}
    for anno in tqdm(object_annos.values()):
        for synset in anno['attribute_synsets']:
            if synset not in freqs:
                freqs[synset] = 1
            else:
                freqs[synset] += 1
    return freqs


def main():
    const = VisualGenomeConstants()
    io.mkdir_if_not_exists(const.proc_dir,recursive=True)

    print('Loading object_annos.json ...')
    object_annos = io.load_json_object(const.object_annos_json)
    
    print('Computing attribute frequencies ...')
    attribute_freqs = compute_attribute_freqs(object_annos)
    print(f'Number of attributes: {len(attribute_freqs)}')
    io.dump_json_object(
        attribute_freqs,
        os.path.join(const.proc_dir,'attribute_freqs.json'))

    print('Computing attribute synset frequencies ...')
    attribute_synset_freqs = compute_attribute_synset_freqs(object_annos)
    print(f'Number of attribute_synsets: {len(attribute_synset_freqs)}')
    io.dump_json_object(
        attribute_synset_freqs,
        os.path.join(const.proc_dir,'attribute_synset_freqs.json'))
        

if __name__=='__main__':
    main()