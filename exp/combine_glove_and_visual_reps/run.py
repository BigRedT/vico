import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from data.glove.constants import GloveConstantsFactory
from . import concat_glove_and_visual_reps


def exp_concat_glove_visual():
    exp_name = 'concat_glove_visual_avg_reps'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/combine_glove_visual_reps')
    exp_const = ExpConstants(exp_name,out_base_dir)

    data_const = Constants()

    # Glove constants
    glove_const = GloveConstantsFactory.create()
    data_const.glove_h5py = glove_const.embeddings_h5py
    data_const.word_to_glove_idx_json = glove_const.word_to_idx_json
    
    # Entity Reps
    data_const.entity_entity_reps_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet/' + \
        'multilabel_resnet_18_fc_center_normalized_adam_entity_entity_reps')
    data_const.entity_entity_reps_npy = os.path.join(
        data_const.entity_entity_reps_dir,
        'reps.npy')
    data_const.entity_wnid_offset_to_idx_json = os.path.join(
        data_const.entity_entity_reps_dir,
        'wnid_to_idx.json')

    data_const.entity_attr_reps_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet/' + \
        'resnet_18_fc_center_normalized_adam_loss_bce_entity_attr_reps')
    data_const.entity_attr_reps_npy = os.path.join(
        data_const.entity_attr_reps_dir,
        'reps.npy')
    data_const.entity_wnid_offset_to_idx_json = os.path.join(
        data_const.entity_attr_reps_dir,
        'wnid_to_idx.json')

    # Attr Reps
    data_const.attr_attr_reps_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes/' + \
        'resnet_18_fc_center_normalized_adam_loss_bce_attr_attr_reps')
    data_const.attr_attr_reps_npy = os.path.join(
        data_const.attr_attr_reps_dir,
        'reps.npy')
    data_const.attr_wnid_to_idx_json = os.path.join(
        data_const.attr_attr_reps_dir,
        'wnid_to_idx.json')

    data_const.attr_entity_reps_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes/' + \
        'multilabel_resnet_18_fc_center_normalized_adam_attr_entity_reps')
    data_const.attr_entity_reps_npy = os.path.join(
        data_const.attr_entity_reps_dir,
        'reps.npy')
    data_const.attr_wnid_to_idx_json = os.path.join(
        data_const.attr_entity_reps_dir,
        'wnid_to_idx.json')

    concat_glove_and_visual_reps.main(exp_const,data_const)





if __name__=='__main__':
    list_exps(globals())
