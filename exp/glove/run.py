import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from data.glove.constants import GloveConstantsFactory
import exp.glove.combine_glove_with_visual_features as \
    combine_glove_with_visual_features


def exp_combine_glove_with_visual_features():
    exp_name = 'concat_glove_and_visual'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/normalized_resnet_embeddings_trained_on_google')
    exp_const = ExpConstants(exp_name,out_base_dir)

    visual_feat_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/normalized_resnet_features_trained_on_google')
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