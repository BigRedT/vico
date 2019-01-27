import os
from tqdm import tqdm

from .constants import VisualGenomeConstants
import utils.io as io


def get_group_annos(object_annos,grouped_attrs,group):
    group_attrs = set(grouped_attrs[group])
    group_annos = {}
    for object_id, anno in tqdm(object_annos.items()):
        selected_attr = []
        for attr in anno['attribute_synsets']:
            if attr in group_attrs:
                selected_attr.append(attr)

        if len(selected_attr)==0:
            continue

        if len(anno['object_synsets'])==0:
            object_synset = '__UNK__'
        else:
            object_synset = anno['object_synsets'][0]

        group_annos[object_id] = {
            'attribute_box': anno['attribute_box'],
            'attribute_synsets': selected_attr,
            'object_synset': object_synset,
        }
    return group_annos


def main():
    const = VisualGenomeConstants()
    print('Loading image to object id ...')
    image_id_to_object_id = io.load_json_object(
        const.image_id_to_object_id_json)

    print('Creating object id to image id ...')
    object_id_to_image_id = {}
    for img_id,obj_ids in tqdm(image_id_to_object_id.items()):
        for obj_id in obj_ids:
            object_id_to_image_id[obj_id] = img_id

    object_id_to_image_id_json = os.path.join(
        const.proc_dir,
        'object_id_to_image_id.json')
    io.dump_json_object(object_id_to_image_id,object_id_to_image_id_json)

    print('Loading object annos ...')
    object_annos = io.load_json_object(const.object_annos_json)

    print('Loading grouped attributes ...')
    grouped_attrs = io.load_json_object(const.attribute_groups_json)
    for group in ['material','shape','color']:
        group_annos = get_group_annos(object_annos,grouped_attrs,group)
        print(group,len(group_annos))
        group_annos_json = os.path.join(
            const.proc_dir,
            f'object_annos_{group}.json')
        io.dump_json_object(group_annos,group_annos_json)


if __name__=='__main__':
    main()