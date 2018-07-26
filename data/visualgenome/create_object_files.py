import os

import utils.io as io
from tqdm import tqdm
from data.visualgenome.constants import VisualGenomeConstants


def get_image_id_to_object_id(objects):
    image_id_to_object_id = {}
    for anno in objects:
        object_ids= []
        for object_anno in anno['objects']:
            object_id = str(object_anno['object_id'])
            object_ids.append(object_id)
        image_id = str(anno['image_id'])
        image_id_to_object_id[image_id] = object_ids
    return image_id_to_object_id


def get_object_annos(objects,attributes,attribute_synsets):
    object_annos = {}
    for anno in tqdm(objects):
        for object_anno in anno['objects']:
            object_id = str(object_anno['object_id'])
            names = object_anno['names']
            object_synsets = object_anno['synsets']
            object_box = [object_anno[v] for v in ['x','y','w','h']]
            object_annos[object_id] = {
                'names': names,
                'object_synsets': object_synsets,
                'object_box': object_box
            }

    for anno in tqdm(attributes):
        for attribute_anno in anno['attributes']:
            object_id = str(attribute_anno['object_id'])
            attribute_box = [attribute_anno[v] for v in ['x','y','w','h']]
            
            if 'attributes' not in attribute_anno:
                attribute_list = []
            else:
                attribute_list = attribute_anno['attributes']
            
            attribute_synset_list = []
            for attribute in attribute_list:
                if attribute not in attribute_synsets:
                    continue
                attribute_synset = attribute_synsets[attribute]
                attribute_synset_list.append(attribute_synset)
            
            if object_id not in object_annos:
                names = attribute_anno['names']
                object_synsets = attribute_anno['synsets']
                object_box = [attribute_anno[v] for v in ['x','y','w','h']]
                object_annos[object_id] = {
                    'names': names,
                    'object_synsets': object_synsets,
                    'object_box': object_box
                }
            object_annos[object_id].update({
                'attribute_box': attribute_box,
                'attributes': attribute_list,
                'attribute_synsets': attribute_synset_list, 
            })

    return object_annos


def main():
    const = VisualGenomeConstants()
    io.mkdir_if_not_exists(const.proc_dir,recursive=True)
    
    print('Loading objects.json ...')
    objects = io.load_json_object(const.objects_json)
    
    print('Loading object_synsets.json ...')
    object_synsets = io.load_json_object(const.object_synsets_json)
    
    print('Creating image_id_to_object_id.json ...')
    image_id_to_object_id = get_image_id_to_object_id(objects)
    io.dump_json_object(
        image_id_to_object_id,
        os.path.join(const.proc_dir,'image_id_to_object_id.json'))

    print('Loading attributes.json ...')
    attributes = io.load_json_object(const.attributes_json)
    
    print('Loading attribute_synsets.json ...')
    attribute_synsets = io.load_json_object(const.attribute_synsets_json)

    print('Creating object_annos.json ...')
    object_annos = get_object_annos(objects,attributes,attribute_synsets)
    io.dump_json_object(
        object_annos,
        os.path.join(const.proc_dir,'object_annos.json'))


if __name__=='__main__':
    main()