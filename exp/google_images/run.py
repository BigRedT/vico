import os
import copy

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
import exp.google_images.cache_resnet_features as cache_resnet_features
from exp.google_images.dataset import GoogleImagesDatasetConstants
from data.semeval_2018_10.constants import SemEval201810Constants
import utils.io as io


def exp_cache_resnet_features():
    exp_name = 'normalized_resnet_features'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images')
    exp_const = ExpConstants(exp_name=exp_name,out_base_dir=out_base_dir)
    exp_const.use_resnet_normalized=True
    exp_const.model_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome/train_resnet_normalized/models/net_249000')
    exp_const.feature_dim = 2048
    
    # Compute vocabulary for semeval2018
    semeval_const = SemEval201810Constants()
    data_const = GoogleImagesDatasetConstants()
    data_const.vocab_json = semeval_const.all_word_freqs

    cache_resnet_features.main(exp_const,data_const)


if __name__=='__main__':
    list_exps(globals())