import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from .models.NET_FILE import NET_CONSTANTS
from .dataset import DATASET_CONSTANTS
from . import train
from . import evaluation


def exp_train():
    exp_name = 'EXP_NAME'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/EXP_GROUP')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 10
    exp_const.model_save_step = 1000
    exp_const.val_step = 1000
    exp_const.num_val_samples = 1000
    exp_const.batch_size = 32
    exp_const.num_epochs = 1000
    exp_const.lr = 0.01
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.optimizer = 'SGD'
    exp_const.subset = {
        'training': 'train',
        'validation': 'val'
    }

    data_const = DATASET_CONSTANTS()

    model_const = Constants()
    model_const.model_num = None
    model_const.net = NET_CONSTANTS()
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    train.main(exp_const,data_const,model_const)


def exp_eval():
    exp_name = 'EXP_NAME'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/EXP_GROUP')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.batch_size = 32
    exp_const.num_workers = 5

    data_const = DATASET_CONSTANTS()
    data_const.subset = 'eval'

    model_const = Constants()
    model_const.model_num = None
    model_const.net = NET_CONSTANTS()
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    evaluation.main(exp_const,data_const,model_const)


if __name__=='__main__':
    list_exps(globals())
