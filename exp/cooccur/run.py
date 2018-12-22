import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from . import fuse_synset_cooccur
from . import word_cooccur
from . import proc_word_cooccur


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


if __name__=='__main__':
    list_exps(globals())
