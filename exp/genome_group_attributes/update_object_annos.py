import os
from tqdm import tqdm

import utils.io as io


def main(exp_const,data_const):
    print('Loading object annos ...')
    object_annos = io.load_json_object(data_const.object_annos_json)

    print('Loading predicted attributes ...')
    pred_attrs = io.load_json_object(data_const.pred_attrs_json)

    for object_id, pred_attrs_ in tqdm(pred_attrs.items()):
        color_attrs = set([attr for attr,prob in pred_attrs_['color']])
        shape_attrs = set([attr for attr,prob in pred_attrs_['shape']])
        all_attrs = set(object_annos[object_id]['attribute_synsets'])
        all_attrs = all_attrs.union(color_attrs)
        all_attrs = all_attrs.union(shape_attrs)
        object_annos[object_id]['attribute_synsets'] = list(all_attrs)

    updated_object_annos_json = os.path.join(
        exp_const.exp_dir,
        'updated_object_annos.json')
    io.dump_json_object(object_annos,updated_object_annos_json)
