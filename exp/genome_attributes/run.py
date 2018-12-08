import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from .models.resnet_normalized import ResnetNormalizedConstants
from .dataset import GenomeAttributesDatasetConstants
from . import train
from . import attr_attr_reps
from . import attr_entity_reps
from . import find_nn


def exp_train():
    exp_name = 'resnet_18_fc_center_normalized_adam_loss_bce'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 10
    exp_const.model_save_step = 10000
    exp_const.vis_step = 5000
    # exp_const.val_step = 1000
    # exp_const.num_val_samples = 1000
    exp_const.batch_size = 10 # 10 for resnet 18; 5 for resnet 50
    exp_const.num_epochs = 20
    exp_const.lr = 0.1
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.optimizer = 'Adam'
    exp_const.margin = 0.5

    data_const = GenomeAttributesDatasetConstants()

    model_const = Constants()
    model_const.model_num = None
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 6497
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    train.main(exp_const,data_const,model_const)


def exp_compute_attr_attr_reps():
    exp_name = 'resnet_18_fc_center_normalized_adam_loss_bce_attr_attr_reps'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 10
    exp_const.num_workers = 5

    data_const = GenomeAttributesDatasetConstants()

    model_const = Constants()
    model_const.model_num = 110000
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 6497
    model_const.net_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes/resnet_18_fc_center_normalized_adam_loss_bce/' + \
        f'models/net_{model_const.model_num}')

    attr_attr_reps.main(exp_const,data_const,model_const)


def exp_compute_attr_entity_reps():
    exp_name = 'multilabel_resnet_18_normalized_adam_attr_entity_reps'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 10
    exp_const.num_workers = 5

    data_const = GenomeAttributesDatasetConstants()

    model_const = Constants()
    model_const.model_num = 930000
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 21841
    model_const.net_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet/multilabel_resnet_18_normalized_adam/' + \
        f'models/net_{model_const.model_num}')

    attr_attr_reps.main(exp_const,data_const,model_const)


def exp_find_nn():
    exp_name = 'resnet_18_fc_center_normalized_adam_loss_bce_attr_attr_reps'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.num_nbrs = 5
    exp_const.min_freq = 100

    data_const = Constants()
    data_const.reps_npy = os.path.join(exp_const.exp_dir,'reps.npy')
    data_const.wnid_to_idx_json = os.path.join(
        exp_const.exp_dir,
        'wnid_to_idx.json')
    data_const.num_imgs_per_class_npy = os.path.join(
        exp_const.exp_dir,
        'num_imgs_per_class.npy')
    data_const.knn_html = os.path.join(
        exp_const.exp_dir,
        'reps_knn.html')

    find_nn.main(exp_const,data_const)

# def exp_eval():
#     exp_name = 'EXP_NAME'
#     out_base_dir = os.path.join(
#         os.getcwd(),
#         'symlinks/exp/EXP_GROUP')
#     exp_const = ExpConstants(exp_name,out_base_dir)
#     exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
#     exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
#     exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
#     exp_const.batch_size = 32
#     exp_const.num_workers = 5

#     data_const = DATASET_CONSTANTS()
#     data_const.subset = 'eval'

#     model_const = Constants()
#     model_const.model_num = None
#     model_const.net = NET_CONSTANTS()
#     model_const.net_path = os.path.join(
#         exp_const.model_dir,
#         f'net_{model_const.model_num}')

#     evaluation.main(exp_const,data_const,model_const)


if __name__=='__main__':
    list_exps(globals())
