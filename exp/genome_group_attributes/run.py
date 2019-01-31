import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
import utils.io as io
from .dataset import GenomeAttributesDatasetConstants
from .models.resnet_object import ResnetConstants
from .dataset import GenomeAttributesDatasetConstants
from .dataset_full import GenomeAttributesDatasetConstants as \
    GenomeAttributesDatasetConstantsFull
from . import extract_attributes
from . import train
from . import update_object_annos
from .vis import vis_pred_attributes


def exp_train():
    group = 'color'
    exp_name = f'resnet_34_object_pos_{group}'
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
    data_const.group = group
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


def exp_extract():
    exp_name = 'resnet_34_object_pos_extract_attrs'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_group_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.batch_size = 10 # 10 for resnet 18; 5 for resnet 50
    exp_const.num_workers = 10
    
    data_const = GenomeAttributesDatasetConstantsFull()
    attr_groups = io.load_json_object(data_const.attribute_groups_json)
    num_objects = len(io.load_json_object(data_const.object_synset_freqs_json))

    model_const = Constants()
    model_const.shape_model_num = 28000
    model_const.material_model_num = 28000
    model_const.color_model_num = 28000
    for group in data_const.groups:
        num_classes = len(attr_groups[group])
        net_const = ResnetConstants()
        net_const.num_layers = 34
        net_const.num_classes = num_classes
        net_const.num_objects = num_objects
        setattr(model_const,f'{group}_net',net_const)
        model_num = getattr(model_const,f'{group}_model_num')
        net_path = os.path.join(
            os.getcwd(),
            'symlinks/exp/genome_group_attributes/' + \
            f'resnet_34_object_pos_{group}/models/' + \
            f'net_{model_num}')
        setattr(model_const,f'{group}_net_path',net_path)

    extract_attributes.main(exp_const,data_const,model_const)


def exp_vis_pred_attributes():
    exp_name = 'resnet_34_object_pos_extract_attrs'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_group_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.batch_size = 2
    exp_const.num_workers = 10
    exp_const.num_samples = 100

    data_const = GenomeAttributesDatasetConstantsFull()
    data_const.pred_attrs_json = os.path.join(
        exp_const.exp_dir,
        'pred_attrs.json')

    vis_pred_attributes.main(exp_const,data_const)


def exp_update_object_annos():
    exp_name = 'resnet_34_object_pos_extract_attrs'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_group_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)

    data_const = GenomeAttributesDatasetConstantsFull()
    data_const.pred_attrs_json = os.path.join(
        exp_const.exp_dir,
        'pred_attrs.json')

    update_object_annos.main(exp_const,data_const)


if __name__=='__main__':
    list_exps(globals())
