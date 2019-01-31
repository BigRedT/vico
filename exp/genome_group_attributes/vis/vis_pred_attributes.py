import os
import torch
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

import utils.io as io
from utils.html_writer import HtmlWriter
from ..dataset_full import GenomeAttributesDataset


def pred_list_to_string_list(l):
    str_list = []
    for l_ in l:
        s = l_[0] # synset
        p = str(round(l_[1],2))
        str_list.append(f'{s} ({p})')
    
    return str_list


def main(exp_const,data_const):
    vis_dir = os.path.join(exp_const.vis_dir,'pred_attributes')
    io.mkdir_if_not_exists(vis_dir,recursive=True)

    print('Loading predicted attributes ...')
    pred_attrs = io.load_json_object(data_const.pred_attrs_json)

    print('Creating dataloader ...')
    dataset = GenomeAttributesDataset(data_const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets',
        'object_words','attribute_words'])
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    html_filename = os.path.join(vis_dir,'vis.html')
    html_writer = HtmlWriter(html_filename)
    col_dict = {
        0: 'Image',
        1: 'Object',
        2: 'GT Attrs',
        3: 'Material',
        4: 'Shape',
        5: 'Color',
    }
    html_writer.add_element(col_dict)

    for it,data in enumerate(tqdm(dataloader)):
        if it > exp_const.num_samples:
            break

        if data is None:
            continue

        images = data['regions'].numpy()
        B = images.shape[0]
        b = random.randint(0,B-1)
        image = images[b]
        object_synsets = data['object_synsets'][b]
        gt_attribute_synsets = data['attribute_synsets'][b]
        object_id = data['object_ids'][b]
        if object_id in pred_attrs:
            pred_attrs_ = pred_attrs[object_id]
            pred_color = pred_list_to_string_list(pred_attrs_['color'])
            pred_material = pred_list_to_string_list(pred_attrs_['material'])
            pred_shape = pred_list_to_string_list(pred_attrs_['shape'])
            pred_gt = pred_attrs_['gt']
        else:
            pred_material = []
            pred_shape = []
            pred_gt = []
            #continue

        image_name = f'{object_id}.png'
        image_path = os.path.join(vis_dir,image_name)
        image = Image.fromarray(image)
        image.save(image_path)
        
        col_dict = {
            0: html_writer.image_tag(image_name),
            1: object_synsets,
            2: gt_attribute_synsets,
            3: pred_material,
            4: pred_shape,
            5: pred_color,
        }

        #import pdb; pdb.set_trace()
        html_writer.add_element(col_dict)

    html_writer.close()