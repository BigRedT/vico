import os

import utils.io as io
from tqdm import tqdm
from data.visualgenome.constants import VisualGenomeConstants


def compute_object_freqs(object_annos):
    freqs = {}
    for anno in tqdm(object_annos.values()):
        for synset in anno['object_synsets']:
            object_name = synset.split('.')[0]
            if object_name not in freqs:
                freqs[object_name] = 1
            else:
                freqs[object_name] += 1
    return freqs
        

def compute_object_synset_freqs(object_annos):
    freqs = {}
    for anno in tqdm(object_annos.values()):
        for synset in anno['object_synsets']:
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
    
    print('Computing object frequencies ...')
    object_freqs = compute_object_freqs(object_annos)
    print(f'Number of objects: {len(object_freqs)}')
    io.dump_json_object(
        object_freqs,
        os.path.join(const.proc_dir,'object_freqs.json'))

    print('Computing object synset frequencies ...')
    object_synset_freqs = compute_object_synset_freqs(object_annos)
    print(f'Number of object_synsets: {len(object_synset_freqs)}')
    io.dump_json_object(
        object_synset_freqs,
        os.path.join(const.proc_dir,'object_synset_freqs.json'))
    
        
if __name__=='__main__':
    main()