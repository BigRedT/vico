import os
from PIL import Image

from ..dataset import GenomeAttributesDataset
import utils.io as io
from utils.html_writer import HtmlWriter


def main(exp_const,data_const):
    print('Loading clustering data ...')
    cluster_ids = io.load_json_object(data_const.cluster_ids_json)
    cluster_id_to_feat_ids = io.load_json_object(
        data_const.cluster_id_to_feat_ids_json)
    object_ids = io.load_json_object(data_const.object_ids_json)
    image_ids = io.load_json_object(data_const.image_ids_json)

    print('Loading dataset ...')
    dataset = GenomeAttributesDataset(data_const)

    vis_dir = os.path.join(exp_const.vis_dir,'clusters')
    io.mkdir_if_not_exists(vis_dir,recursive=True)

    html_writer = HtmlWriter(os.path.join(vis_dir,'vis.html'))

    for i, (cid,feat_ids) in enumerate(cluster_id_to_feat_ids.items()):
        print(i)

        if i > exp_const.num_clusters_to_vis:
            break

        cluster_dir = os.path.join(vis_dir,cid)
        io.mkdir_if_not_exists(cluster_dir)

        col_dict = {0: cid}

        for k,j in enumerate(feat_ids[:20]):
            image_id = image_ids[j]
            object_id = object_ids[j]
            img,_ = dataset.get_image(image_id)
            region = dataset.crop_region(
                img,
                dataset.object_annos[object_id]['attribute_box'],
                0)
            region.save(os.path.join(cluster_dir,str(k)+'.png'))
            
            col_dict[k+1] = html_writer.image_tag(
                f'{cid}/{str(k)}.png',
                height=100,
                width=100)

        html_writer.add_element(col_dict)

    html_writer.close()
            
        

