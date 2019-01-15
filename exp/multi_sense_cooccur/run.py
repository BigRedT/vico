import os

from exp.experimenter import *
import utils.io as io
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from . import merge_cooccur
from .dataset import MultiSenseCooccurDatasetConstants
from .models.logbilinear import LogBilinearConstants
from . import train
from . import train_w_neg
from . import extract_embeddings
from . import find_nn


def exp_merge_cooccur():
    exp_name = 'imagenet_genome_gt'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur')
    exp_const = ExpConstants(exp_name,out_base_dir)

    data_const = Constants()
    cooccur_paths = {
        'syn': 'wordnet/syn_cooccur/word_cooccur.json',
        'attr_attr': 'genome_attributes/gt_attr_attr_cooccur/word_cooccur.json',
        'obj_attr': 'genome_attributes/gt_obj_attr_cooccur/word_cooccur.json',
        'obj_hyp': 'imagenet/gt_obj_hyp_cooccur/word_cooccur.json',
        'context': 'genome_attributes/gt_context_cooccur/word_cooccur.json',
    }
    symlink_exp = os.path.join(os.getcwd(),'symlinks/exp')
    data_const.cooccur_paths = {
        k: os.path.join(symlink_exp,v) for k,v in cooccur_paths.items()}

    merge_cooccur.main(exp_const,data_const)


def exp_train():
    exp_name = 'all_cooccur_dim_100_add_0_batch_1000_neg_fx' #_weight_decay'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 10
    exp_const.model_save_step = 10000
    exp_const.batch_size = 1000
    exp_const.num_epochs = 10
    exp_const.lr = 0.01 
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.optimizer = 'Adam'
    exp_const.weight_decay = 1e-4
    exp_const.cooccur_weights = {
        'syn': 1,
        'attr_attr': 1,
        'obj_attr': 1,
        'obj_hyp': 1,
        'context': 1,
    }
    
    data_const = MultiSenseCooccurDatasetConstants()
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt/merged_cooccur.csv')

    model_const = Constants()
    model_const.model_num = None
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 78416
    model_const.net.embed_dims = 100
    model_const.net.two_embedding_layers = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    #train.main(exp_const,data_const,model_const)
    train_w_neg.main(exp_const,data_const,model_const)


def exp_extract_embeddings():
    exp_name = 'all_cooccur_dim_100_add_0_batch_1000_wo_neg'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.cooccur_types = [
        'syn',
        'attr_attr',
        'obj_attr',
        'obj_hyp',
        'context']

    data_const = MultiSenseCooccurDatasetConstants()
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt/merged_cooccur.csv')

    model_const = Constants()
    model_const.model_num = 110000
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 78416
    model_const.net.embed_dims = 100
    model_const.net.two_embedding_layers = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    extract_embeddings.main(exp_const,data_const,model_const)


def exp_find_nn():
    exp_name = 'all_cooccur_dim_100_add_0_batch_1000_neg'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.num_nbrs = 10
    exp_const.use_cosine = False
    exp_const.thresh = [
        10000,
        100000,
        100000,
        100000,
        100000,
    ]
    exp_const.cooccur_types = [
        'syn',
        'attr_attr',
        'obj_attr',
        'obj_hyp',
        'context']

    data_const = Constants()
    data_const.embeddings_npy = os.path.join(
        exp_const.exp_dir,
        'visual_embeddings.npy')
    data_const.word_to_idx_json = os.path.join(
        exp_const.exp_dir,
        'word_to_idx.json')
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt/merged_cooccur.csv')

    find_nn.main(exp_const,data_const)

if __name__=='__main__':
    list_exps(globals())