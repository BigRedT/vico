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
from . import extract_embeddings, extract_embeddings_xformed
from . import find_nn
from . import concat_with_glove
from data.glove.constants import GloveConstantsFactory
from . import synset_to_word_cooccur
from .vis import bias_vs_self_count #as bias_vs_self_count


def exp_synset_to_word_cooccur():
    paths = [
        ['imagenet','gt_obj_hyp_cooccur_self'],
        ['genome_attributes','gt_attr_attr_cooccur_self'],
        ['genome_attributes','gt_obj_attr_cooccur_self'],
        ['genome_attributes','gt_context_cooccur_self'],
    ]
    for dataset,exp_name in paths:
        out_base_dir = os.path.join(
            os.getcwd(),
            f'symlinks/exp/{dataset}')
        exp_const = ExpConstants(exp_name,out_base_dir)

        data_const = Constants()
        data_const.synset_cooccur_json = os.path.join(
            exp_const.exp_dir,
            'synset_cooccur.json')
        data_const.word_cooccur_json = os.path.join(
            exp_const.exp_dir,
            'word_cooccur.json')

        synset_to_word_cooccur.main(exp_const,data_const)


def exp_merge_cooccur():
    exp_name = 'imagenet_genome_gt'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.normalize = False

    data_const = Constants()
    cooccur_paths = {
        'syn': 'wordnet/syn_cooccur_self/word_cooccur.json',
        'attr_attr': 'genome_attributes/gt_attr_attr_cooccur_self/word_cooccur.json',
        'obj_attr': 'genome_attributes/gt_obj_attr_cooccur_self/word_cooccur.json',
        'obj_hyp': 'imagenet/gt_obj_hyp_cooccur_self/word_cooccur.json',
        'context': 'genome_attributes/gt_context_cooccur_self/word_cooccur.json',
    }
    symlink_exp = os.path.join(os.getcwd(),'symlinks/exp')
    data_const.cooccur_paths = {
        k: os.path.join(symlink_exp,v) for k,v in cooccur_paths.items()}
    data_const.merged_cooccur_csv = os.path.join(
        exp_const.exp_dir,
        'merged_cooccur_self.csv') # 'merged_cooccur.csv'
    merge_cooccur.main(exp_const,data_const)


def exp_train():
    exp_name = 'dim_200_neg_bias_nonlinear_4_tanh_adagrad'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
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
    exp_const.optimizer = 'Adagrad'
    exp_const.weight_decay = 0*1e-4
    exp_const.cooccur_weights = {
        # 'syn': 0,
        'attr_attr': 1,
        'obj_attr': 1,
        'obj_hyp': 1,
        'context': 1,
    }
    exp_const.use_neg = True
    
    data_const = MultiSenseCooccurDatasetConstants()
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt/merged_cooccur_self.csv')
    data_const.use_self_count = True

    model_const = Constants()
    model_const.model_num = None
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 93553 #91138 #78416
    model_const.net.embed_dims = 200
    model_const.net.two_embedding_layers = False
    model_const.net.xform_type = 'nonlinear'
    model_const.net.xform_num_layers = 4
    model_const.net.use_bias = True
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    #train.main(exp_const,data_const,model_const)
    train_w_neg.main(exp_const,data_const,model_const)


def exp_extract_embeddings():
    exp_name = 'dim_200_neg_bias_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.cooccur_types = [
        #'syn',
        'attr_attr',
        'obj_attr',
        'obj_hyp',
        'context']

    data_const = MultiSenseCooccurDatasetConstants()
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt/merged_cooccur_self.csv')

    model_const = Constants()
    model_const.model_num = 100000
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 93553
    model_const.net.embed_dims = 200
    model_const.net.two_embedding_layers = False
    model_const.net.xform_type = 'linear'
    model_const.net.xform_num_layers = None
    model_const.net.use_bias = True
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    extract_embeddings.main(exp_const,data_const,model_const)
    extract_embeddings_xformed.main(exp_const,data_const,model_const)


def exp_find_nn():
    exp_name = 'dim_200_neg_bias_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_nbrs = 20
    exp_const.use_cosine = False
    exp_const.thresh = [
        #10000,
        100000,
        100000,
        100000,
        100000,
    ]
    exp_const.cooccur_types = [
        #'syn',
        'attr_attr',
        'obj_attr',
        'obj_hyp',
        'context'
    ]

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
        'imagenet_genome_gt/merged_cooccur_self.csv')

    model_const = Constants()
    model_const.model_num = 100000
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 93553
    model_const.net.embed_dims = 200
    model_const.net.two_embedding_layers = False
    model_const.net.xform_type = 'linear'
    model_const.net.xform_num_layers = None
    model_const.net.use_bias = True
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    find_nn.main(exp_const,data_const,model_const)


def exp_concat_with_glove():
    exp_name = 'concat_with_glove_300' # alt. xformed_
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/' + \
        'effect_of_xforms/' + \
        'dim_200_neg_bias_linear')
    exp_const = ExpConstants(exp_name,out_base_dir)

    visual_embed_dir = exp_const.out_base_dir
    data_const = Constants()
    data_const.visual_word_to_idx = os.path.join(
        visual_embed_dir,
        'word_to_idx.json')
    data_const.visual_embeddings_npy = os.path.join(
        visual_embed_dir,
        'visual_embeddings.npy') # alt. _xformed.npy
    glove_const = GloveConstantsFactory.create(dim='300')
    data_const.glove_idx = glove_const.word_to_idx_json
    data_const.glove_h5py = glove_const.embeddings_h5py

    concat_with_glove.main(exp_const,data_const)


def exp_vis_bias_vs_self_count():
    exp_name = 'dim_200_neg_bias_select'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')

    data_const = MultiSenseCooccurDatasetConstants()
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt/merged_cooccur_self.csv')

    model_const = Constants()
    model_const.model_num = 100000
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 93553
    model_const.net.embed_dims = 200
    model_const.net.two_embedding_layers = False
    model_const.net.xform_type = 'select'
    model_const.net.xform_num_layers = None
    model_const.net.use_bias = True
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')
    
    bias_vs_self_count.main(exp_const,data_const,model_const)


if __name__=='__main__':
    list_exps(globals())