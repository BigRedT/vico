import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from .dataset import GenomeAttributesDatasetConstants
from . import extract_lemniscate_feats
from . import cluster
from . import knn
from .vis import vis_clusters
from .vis import vis_knn


def exp_extract_feats():
    exp_name = 'lemniscate_feats'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_clustering')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.batch_size = 32
    exp_const.num_workers = 10

    data_const = GenomeAttributesDatasetConstants()

    model_const = Constants()
    model_const.net_path = os.path.join(
        os.getcwd(),
        f'symlinks/exp/lemniscate_resnet/resnet50')

    extract_lemniscate_feats.main(exp_const,data_const,model_const)


def exp_cluster():
    exp_name = 'lemniscate_feats'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_clustering')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.n_clusters = 1000
    exp_const.batch_size = 10000
    exp_const.num_epochs = 5

    data_const = GenomeAttributesDatasetConstants()
    data_const.feats_npy = os.path.join(exp_const.exp_dir,'feats.npy')
    data_const.object_ids_json = os.path.join(
        exp_const.exp_dir,
        'object_ids.json')
    data_const.image_ids_json = os.path.join(
        exp_const.exp_dir,
        'image_ids.json')

    cluster.main(exp_const,data_const)


def exp_vis_clusters():
    exp_name = 'lemniscate_feats'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_clustering')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.num_clusters_to_vis = 20

    data_const = GenomeAttributesDatasetConstants()
    data_const.object_ids_json = os.path.join(
        exp_const.exp_dir,
        'object_ids.json')
    data_const.image_ids_json = os.path.join(
        exp_const.exp_dir,
        'image_ids.json')
    data_const.cluster_ids_json = os.path.join(
        exp_const.exp_dir,
        'cluster_ids.json')
    data_const.cluster_id_to_feat_ids_json = os.path.join(
        exp_const.exp_dir,
        'cluster_id_to_feat_ids.json')

    vis_clusters.main(exp_const,data_const)
    

def exp_knn():
    exp_name = 'lemniscate_feats'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_clustering')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.k = 5

    data_const = Constants()
    data_const.feats_npy = os.path.join(exp_const.exp_dir,'feats.npy')

    knn.main(exp_const,data_const)


def exp_vis_knn():
    exp_name = 'lemniscate_feats'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_clustering')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.num_nbrs_to_vis = 20

    data_const = GenomeAttributesDatasetConstants()
    data_const.object_ids_json = os.path.join(
        exp_const.exp_dir,
        'object_ids.json')
    data_const.image_ids_json = os.path.join(
        exp_const.exp_dir,
        'image_ids.json')
    data_const.knn_ids_npy = os.path.join(
        exp_const.exp_dir,
        'knn_ids.npy')

    vis_knn.main(exp_const,data_const)


if __name__=='__main__':
    list_exps(globals())
