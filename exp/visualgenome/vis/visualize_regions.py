import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

import utils.io as io
from utils.html_writer import HtmlWriter
from exp.visualgenome.image_regions_dataset import ImageRegionsDataset


def main(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)

    print('Creating dataloader ...')
    dataset = ImageRegionsDataset(data_const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets','object_ids'])
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=exp_const.shuffle,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    print('Creating html writer ...')
    filename = os.path.join(
        exp_const.exp_dir,
        'vis.html')
    html_writer = HtmlWriter(filename)

    print('Writing table header ...')
    col_dict = {
        0: 'Image Id',
        1: 'Object Id',
        2: 'Region',
        3: 'Objects',
        4: 'Attributes'
    }
    html_writer.add_element(col_dict)
    for i,data in enumerate(tqdm(dataloader)):
        if i==exp_const.num_batches:
            break
        if data is None:
            continue
        num_regions = len(data['regions'])
        for j in range(num_regions):
            region_id = data['object_ids'][j]
            region_name = os.path.join(
                exp_const.exp_dir,
                f'{region_id}.jpg')
            region_img = Image.fromarray(data['regions'][j].astype(np.uint8))
            region_img.save(region_name,'JPEG')
            col_dict = {
                0: data['image_ids'][j],
                1: data['object_ids'][j],
                2: html_writer.image_tag(f'{region_id}.jpg'),
                3: data['object_synsets'][j],
                4: data['attribute_synsets'][j],
            }
            html_writer.add_element(col_dict)
    html_writer.close()
    
        

        
    

