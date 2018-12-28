import os

from exp.experimenter import *
import utils.io as io
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from . import fuse_synset_cooccur
from . import word_cooccur
from . import proc_word_cooccur
from .dataset import CooccurDatasetConstants
from .models.logbilinear import LogBilinearConstants
from . import train
from . import extract_embeddings
from . import find_nn
from . import concat_with_glove
from data.glove.constants import GloveConstantsFactory


def exp_fuse_synset_cooccur():
    exp_name = 'imagenet_genome_gt'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur')
    exp_const = ExpConstants(exp_name,out_base_dir)

    data_const = Constants()
    data_const.imagenet_synset_cooccur_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet/gt_cooccur/synset_cooccur.json')
    data_const.genome_synset_cooccur_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes/gt_cooccur/synset_cooccur.json')

    fuse_synset_cooccur.main(exp_const,data_const)


def exp_synset_to_word_cooccur():
    exp_name = 'imagenet_genome_gt'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur')
    exp_const = ExpConstants(exp_name,out_base_dir)

    data_const = Constants()
    data_const.synset_cooccur_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt/fused_synset_cooccur.json')
    data_const.word_cooccur_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt/fused_word_cooccur.json')

    word_cooccur.main(exp_const,data_const)


def exp_proc_word_cooccur():
    exp_name = 'imagenet_genome_gt'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur')
    exp_const = ExpConstants(exp_name,out_base_dir)

    data_const = Constants()
    data_const.word_cooccur_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt/fused_word_cooccur.json')
    data_const.proc_word_cooccur_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt/proc_fused_word_cooccur.json')

    proc_word_cooccur.main(exp_const,data_const)


def exp_train():
    exp_name = 'training_no_fx_self_count_dim_50_single_embed'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 100
    exp_const.model_save_step = 10000
    exp_const.batch_size = 1000
    exp_const.num_epochs = 50
    exp_const.lr = 0.001
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.optimizer = 'Adagrad'
    
    data_const = CooccurDatasetConstants()
    data_const.cooccur_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt/fused_word_cooccur.json')
    cooccur = io.load_json_object(data_const.cooccur_json)
    num_words = len(cooccur)

    model_const = Constants()
    model_const.model_num = 280000
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = num_words
    model_const.net.embed_dims = 50
    model_const.net.two_embedding_layers = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    train.main(exp_const,data_const,model_const)


def exp_extract_embeddings():
    exp_name = 'training_no_fx_self_count_dim_50_single_embed'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')

    data_const = CooccurDatasetConstants()
    data_const.cooccur_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt/fused_word_cooccur.json')
    cooccur = io.load_json_object(data_const.cooccur_json)
    num_words = len(cooccur)

    model_const = Constants()
    model_const.model_num = 420000
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = num_words
    model_const.net.embed_dims = 50
    model_const.net.two_embedding_layers = True
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    extract_embeddings.main(exp_const,data_const,model_const)


def exp_find_nn():
    exp_name = 'training_no_fx_self_count_dim_50_single_embed'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.num_nbrs = 10
    exp_const.min_freq = 10000

    data_const = Constants()
    data_const.embeddings_npy = os.path.join(
        exp_const.exp_dir,
        'visual_embeddings.npy')
    data_const.word_to_idx_json = os.path.join(
        exp_const.exp_dir,
        'word_to_idx.json')
    data_const.knn_html = os.path.join(
        exp_const.exp_dir,
        'visual_embeddings_knn.html')
    data_const.cooccur_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt/fused_word_cooccur.json')

    find_nn.main(exp_const,data_const)


def exp_concat_with_glove():
    exp_name = 'concat_with_glove_300'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cooccur/imagenet_genome_gt/training_no_fx_self_count_dim_50_single_embed')
    exp_const = ExpConstants(exp_name,out_base_dir)

    visual_embed_dir = exp_const.out_base_dir
    data_const = Constants()
    data_const.visual_word_to_idx = os.path.join(
        visual_embed_dir,
        'word_to_idx.json')
    data_const.visual_embeddings_npy = os.path.join(
        visual_embed_dir,
        'visual_embeddings.npy')
    glove_const = GloveConstantsFactory.create(dim='300')
    data_const.glove_idx = glove_const.word_to_idx_json
    data_const.glove_h5py = glove_const.embeddings_h5py

    concat_with_glove.main(exp_const,data_const)

if __name__=='__main__':
    list_exps(globals())
