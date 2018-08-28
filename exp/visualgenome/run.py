import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from data.glove.constants import GloveConstantsFactory
from data.visualgenome.constants import VisualGenomeConstants
from exp.visualgenome.image_regions_dataset import ImageRegionsDatasetConstants
import exp.visualgenome.vis.visualize_regions as visualize_regions
import exp.visualgenome.cache_resnet_features as cache_resnet_features
import exp.visualgenome.collect_visual_features as collect_visual_features
import exp.visualgenome.combine_glove_with_visual_features as \
    combine_glove_with_visual_features

def exp_visualize_regions():
    exp_name = 'region_visualization'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.num_batches = 10
    exp_const.batch_size = 5
    exp_const.shuffle = False
    exp_const.num_workers = 5
    
    data_const = ImageRegionsDatasetConstants()

    visualize_regions.main(exp_const,data_const)


def exp_cache_resnet_features():
    exp_name = 'resnet_features'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 5
    exp_const.num_workers = 5
    exp_const.feature_dim = 2048 # 2208 for densenet161, 1024 for densent 121, 2048 for resnet152

    data_const = ImageRegionsDatasetConstants()
    data_const.imagenet_labels_json = os.path.join(
        os.getcwd(),
        'symlinks/data/imagenet/proc/labels.json')

    cache_resnet_features.main(exp_const,data_const)


def exp_collect_visual_features():
    exp_name = 'pretrained_resnet_embeddings'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome')
    exp_const = ExpConstants(exp_name,out_base_dir)
    
    data_const = VisualGenomeConstants()
    data_const.visual_features_h5py = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome/resnet_features/' + \
        'visual_features.h5py')
    data_const.object_id_to_features_idx_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome/resnet_features/' + \
        'object_id_to_features_idx.json')
    
    collect_visual_features.main(exp_const,data_const)


def exp_combine_glove_with_visual_features():
    exp_name = 'concat_glove_and_visual'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome/pretrained_resnet_embeddings')
    exp_const = ExpConstants(exp_name,out_base_dir)

    visual_feat_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome/pretrained_resnet_embeddings')
    data_const = Constants()
    data_const.visual_features_idx = os.path.join(
        visual_feat_dir,
        'word_to_idx.json')
    data_const.visual_features_h5py = os.path.join(
        visual_feat_dir,
        'word_features.h5py')
    glove_const = GloveConstantsFactory.create()
    data_const.glove_idx = glove_const.word_to_idx_json
    data_const.glove_h5py = glove_const.embeddings_h5py

    combine_glove_with_visual_features.main(exp_const,data_const)


if __name__=='__main__':
    list_exps(globals())
