import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from .dataset import ImagenetDatasetConstants
from . import create_gt_obj_hyp_cooccur


def exp_create_gt_obj_hyp_cooccur():
    exp_name = 'gt_obj_hyp_cooccur'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 32
    exp_const.num_workers = 5

    data_const = ImagenetDatasetConstants()

    create_gt_obj_hyp_cooccur.main(exp_const,data_const)


if __name__=='__main__':
    list_exps(globals())
