import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from .dataset import GenomeAttributesDatasetConstants
from .models.resnet_object import ResnetConstants
from .dataset import GenomeAttributesDatasetConstants
from . import train
import utils.io as io
import matplotlib


def exp_train():
    exp_name = 'resnet_34_object_material'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_group_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 10
    exp_const.model_save_step = 2000
    exp_const.vis_step = 300
    # exp_const.val_step = 1000
    # exp_const.num_val_samples = 1000
    exp_const.batch_size = 256 # 10 for resnet 18; 5 for resnet 50
    exp_const.num_epochs = 1000
    exp_const.lr = 0.01
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.optimizer = 'Adam'

    data_const = GenomeAttributesDatasetConstants()
    data_const.group = 'material'
    attr_groups = io.load_json_object(data_const.attribute_groups_json)
    num_classes = len(attr_groups[data_const.group])
    num_objects = len(io.load_json_object(data_const.object_synset_freqs_json))

    model_const = Constants()
    model_const.model_num = None
    model_const.net = ResnetConstants()
    model_const.net.num_layers = 34
    model_const.net.num_classes = num_classes
    model_const.net.num_objects = num_objects
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    train.main(exp_const,data_const,model_const)



if __name__=='__main__':
    list_exps(globals())
