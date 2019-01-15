import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from data.visualgenome.constants import VisualGenomeConstants
from .models.resnet_normalized import ResnetNormalizedConstants
from .dataset import GenomeAttributesDatasetConstants
from . import train
from . import train_resnet
from . import attr_attr_reps
from . import attr_entity_reps
from . import find_nn
from . import class_weights
from . import attr_cooccur
from . import create_gt_cooccur
from . import create_gt_obj_attr_cooccur
from . import create_gt_attr_attr_cooccur
from . import create_gt_context_cooccur
from .vis import vis_pred


def exp_create_gt_cooccur():
    exp_name = 'gt_cooccur'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 32
    exp_const.num_workers = 5

    data_const = GenomeAttributesDatasetConstants()

    create_gt_cooccur.main(exp_const,data_const)


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


def exp_compute_class_weights():
    exp_name = 'class_weights'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 10
    exp_const.num_workers = 5

    data_const = GenomeAttributesDatasetConstants()

    class_weights.main(exp_const,data_const)


def exp_train_resnet():
    exp_name = 'resnet_ce'
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
    exp_const.lr = 0.01
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.optimizer = 'Adam'
    exp_const.margin = 0.5
    #exp_const.p = 1

    data_const = GenomeAttributesDatasetConstants()
    data_const.class_weights_npy = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes/class_weights/class_weights.npy')

    model_const = Constants()
    model_const.model_num = None
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 6497
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    train_resnet.main(exp_const,data_const,model_const)


def exp_vis_resnet_pred():
    exp_name = 'resnet_ce'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.batch_size = 10 # 10 for resnet 18; 5 for resnet 50
    exp_const.num_epochs = 20
    exp_const.num_workers = 5
    exp_const.num_nbrs = 20

    data_const = GenomeAttributesDatasetConstants()
    data_const.class_weights_npy = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes/class_weights/class_weights.npy')

    model_const = Constants()
    model_const.model_num = 70000
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 6497
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    vis_pred.main(exp_const,data_const,model_const)


def exp_train():
    exp_name = 'resnet_18_mean_adam_loss_balanced_bce_norm1_relu'
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
    data_const.class_weights_npy = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes/class_weights/class_weights.npy')

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
    exp_name = 'resnet_18_mean_adam_loss_balanced_bce_norm1_relu_attr_attr_reps'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 10
    exp_const.num_workers = 5

    data_const = GenomeAttributesDatasetConstants()

    model_const = Constants()
    model_const.model_num = 210000
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 6497
    model_const.net_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes/resnet_18_mean_adam_loss_balanced_bce_norm1_relu/' + \
        f'models/net_{model_const.model_num}')

    attr_attr_reps.main(exp_const,data_const,model_const)


def exp_compute_attr_entity_reps():
    exp_name = 'multilabel_resnet_18_mean_adam_loss_balanced_bce_norm1_attr_entity_reps'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 10
    exp_const.num_workers = 5

    data_const = GenomeAttributesDatasetConstants()

    model_const = Constants()
    model_const.model_num = 490000
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 21841
    model_const.net_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet/multilabel_resnet_18_mean_adam_loss_balanced_bce_norm1/' + \
        f'models/net_{model_const.model_num}')

    attr_entity_reps.main(exp_const,data_const,model_const)


def exp_find_nn():
    exp_name = 'resnet_18_mean_adam_loss_balanced_bce_norm1_relu_attr_attr_reps'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.num_nbrs = 10
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

    if 'attr_attr' in exp_const.exp_name:
        data_const.reps_npy = os.path.join(exp_const.exp_dir,'classifier_reps.npy')
        data_const.knn_html = os.path.join(
            exp_const.exp_dir,
            'classifier_reps_knn.html')

        find_nn.main(exp_const,data_const)


def exp_compute_attr_cooccur():
    exp_name = 'resnet_18_mean_adam_loss_balanced_bce_norm1_relu_attr_coocur'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 10
    exp_const.num_workers = 5

    data_const = GenomeAttributesDatasetConstants()

    model_const = Constants()
    model_const.model_num = 210000
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 6497
    model_const.net_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes/resnet_18_mean_adam_loss_balanced_bce_norm1_relu/' + \
        f'models/net_{model_const.model_num}')

    attr_cooccur.main(exp_const,data_const,model_const)


if __name__=='__main__':
    list_exps(globals())
