import os
import numpy as np
from tqdm import tqdm
from PIL import Image

from ..dataset import GenomeAttributesDataset
import utils.io as io
from utils.html_writer import HtmlWriter


def main(exp_const,data_const):
    print('Loading clustering data ...')
    knn_ids = np.load(data_const.knn_ids_npy)
    object_ids = io.load_json_object(data_const.object_ids_json)
    image_ids = io.load_json_object(data_const.image_ids_json)

    print('Loading dataset ...')
    dataset = GenomeAttributesDataset(data_const)

    vis_dir = os.path.join(exp_const.vis_dir,'knn')
    io.mkdir_if_not_exists(vis_dir,recursive=True)

    html_writer = HtmlWriter(os.path.join(vis_dir,'vis.html'))

    for i in tqdm(range(exp_const.num_nbrs_to_vis)):
        nbr_dir = os.path.join(vis_dir,str(i))
        io.mkdir_if_not_exists(nbr_dir)

        col_dict = {0: str(i)}
        nbr_ids = [i] + knn_ids[i].tolist()
        for k,j in enumerate(nbr_ids):
            image_id = image_ids[j]
            object_id = object_ids[j]
            img,_ = dataset.get_image(image_id)
            region = dataset.crop_region(
                img,
                dataset.object_annos[object_id]['attribute_box'],
                0)
            region.save(os.path.join(nbr_dir,str(k)+'.png'))
            
            col_dict[k+1] = html_writer.image_tag(
                f'{str(i)}/{str(k)}.png',
                height=100,
                width=100)

        html_writer.add_element(col_dict)

    html_writer.close()
            
        

