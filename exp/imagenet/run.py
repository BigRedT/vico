import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from .models.resnet import ResnetConstants
from .models.resnet_normalized import ResnetNormalizedConstants
from .dataset import ImagenetDatasetConstants
from . import class_weights
from . import train
from . import entity_entity_reps
from . import entity_attr_reps
from . import find_nn
from . import eval as evaluation
from . import create_gt_cooccur


def exp_create_gt_cooccur():
    exp_name = 'gt_cooccur'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 32
    exp_const.num_workers = 5

    data_const = ImagenetDatasetConstants()

    create_gt_cooccur.main(exp_const,data_const)


def exp_compute_class_weights():
    exp_name = 'class_weights'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet')
    exp_const = ExpConstants(exp_name,out_base_dir)

    data_const = ImagenetDatasetConstants()

    class_weights.main(exp_const,data_const)


def exp_train():
    exp_name = 'multilabel_resnet_18_mean_adam_loss_balanced_bce_norm1_relu'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 10
    exp_const.model_save_step = 10000
    exp_const.vis_step = 5000
    # exp_const.val_step = 1000
    # exp_const.num_val_samples = 1000
    exp_const.batch_size = 32
    exp_const.num_epochs = 20
    exp_const.lr = 0.1
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.optimizer = 'Adam'

    data_const = ImagenetDatasetConstants()
    data_const.class_weights_npy = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet/class_weights/class_weights.npy')

    model_const = Constants()
    model_const.model_num = None
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 21841
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    train.main(exp_const,data_const,model_const)


def exp_compute_entity_entity_reps():
    exp_name = 'multilabel_resnet_18_mean_adam_loss_balanced_bce_norm1_relu_entity_entity_reps'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 32
    exp_const.num_workers = 5

    data_const = ImagenetDatasetConstants()

    model_const = Constants()
    model_const.model_num = 700000
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 21841
    model_const.net_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet/multilabel_resnet_18_mean_adam_loss_balanced_bce_norm1_relu/' + \
        f'models/net_{model_const.model_num}')

    entity_entity_reps.main(exp_const,data_const,model_const)


def exp_compute_entity_attr_reps():
    exp_name = 'resnet_18_mean_adam_loss_balanced_bce_norm1_entity_attr_reps'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 64
    exp_const.num_workers = 5

    data_const = ImagenetDatasetConstants()

    model_const = Constants()
    model_const.model_num = 210000
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 6497
    model_const.net_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes/resnet_18_mean_adam_loss_balanced_bce_norm1/' + \
        f'models/net_{model_const.model_num}')

    entity_attr_reps.main(exp_const,data_const,model_const)


def exp_find_nn():
    exp_name = 'resnet_18_mean_adam_loss_balanced_bce_norm1_entity_attr_reps'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.num_nbrs = 10
    exp_const.min_freq = 500

    data_const = Constants()
    data_const.reps_npy = os.path.join(exp_const.exp_dir,'reps.npy')
    data_const.wnid_to_idx_json = os.path.join(
        exp_const.exp_dir,
        'wnid_to_idx.json')
    data_const.num_imgs_per_class_npy = os.path.join(
        exp_const.exp_dir,
        'num_imgs_per_class.npy')
    data_const.wnid_to_words_json = os.path.join(
        os.getcwd(),
        'symlinks/data/imagenet/wnid_to_words.json')
    data_const.knn_html = os.path.join(
        exp_const.exp_dir,
        'reps_knn.html')

    find_nn.main(exp_const,data_const)

    if 'entity_entity' in exp_const.exp_name:
        data_const.reps_npy = os.path.join(exp_const.exp_dir,'classifier_reps.npy')
        data_const.knn_html = os.path.join(
            exp_const.exp_dir,
            'classifier_norm_reps_knn.html')

        find_nn.main(exp_const,data_const)


def exp_eval_classifiers():
    exp_name = 'multilabel_resnet_18_mean_adam_loss_balanced_bce_norm1'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet')
        
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 32
    exp_const.num_workers = 5
    exp_const.num_eval_batches = 2000

    data_const = ImagenetDatasetConstants()

    model_const = Constants()
    model_const.model_num = 1160000
    model_const.net = ResnetNormalizedConstants()
    model_const.net.num_layers = 18
    model_const.net.num_classes = 21841
    model_const.net_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/imagenet/multilabel_resnet_18_mean_adam_loss_balanced_bce_norm1/' + \
        f'models/net_{model_const.model_num}')

    evaluation.main(exp_const,data_const,model_const)


if __name__=='__main__':
    list_exps(globals())
