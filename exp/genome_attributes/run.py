import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from data.visualgenome.constants import VisualGenomeConstants
from .dataset import GenomeAttributesDatasetConstants
from . import create_gt_obj_attr_cooccur
from . import create_gt_attr_attr_cooccur
from . import create_gt_context_cooccur


def exp_create_gt_obj_attr_cooccur():
    exp_name = 'gt_obj_attr_cooccur'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 32
    exp_const.num_workers = 5

    data_const = GenomeAttributesDatasetConstants()

    create_gt_obj_attr_cooccur.main(exp_const,data_const)


def exp_create_gt_attr_attr_cooccur():
    exp_name = 'gt_attr_attr_cooccur'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 32
    exp_const.num_workers = 5

    data_const = GenomeAttributesDatasetConstants()

    create_gt_attr_attr_cooccur.main(exp_const,data_const)


def exp_create_gt_context_cooccur():
    exp_name = 'gt_context_cooccur'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)

    data_const = VisualGenomeConstants()

    create_gt_context_cooccur.main(exp_const,data_const)


if __name__=='__main__':
    list_exps(globals())
