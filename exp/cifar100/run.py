import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from .models.resnet import ResnetConstants
from .models.embed_to_classifier import Embed2ClassConstants
from .dataset import Cifar100DatasetConstants
from . import train
from . import eval as evaluation
from .vis import vis_conf_mat
from .vis import conf_vs_visual_sim


def exp_train():
    #exp_name = 'training_no_fx_self_count_dim_50_single_embed_concat_with_glove_300_sgd_0.01'
    exp_name = 'glove_sgd_0.01'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 100
    exp_const.model_save_step = 1000
    exp_const.val_step = 1000
    #exp_const.num_val_samples = 1000
    exp_const.batch_size = 128
    exp_const.num_epochs = 100
    exp_const.lr = 0.001 #0.01 with adam; finetune with 0.01 sgd
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.optimizer = 'SGD'
    exp_const.subset = {
        'training': 'train',
        'test': 'test'
    }

    data_const = Cifar100DatasetConstants()

    model_const = Constants()
    model_const.model_num = 32000 #, 48000
    model_const.net = ResnetConstants()
    model_const.net.num_layers = 32
    model_const.net.num_classes = 100
    model_const.net.pretrained = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')
    model_const.embed2class = Embed2ClassConstants()
    model_const.embed2class_path = os.path.join(
        exp_const.model_dir,
        f'embed2class_{model_const.model_num}')

    use_glove = True
    if use_glove==True:
        #Glove
        model_const.embed2class.embed_dims = 300
        model_const.embed2class.embed_h5py = os.path.join(
            os.getcwd(),
            'symlinks/data/glove/proc/glove_6B_300d.h5py')
        model_const.embed2class.embed_word_to_idx_json = os.path.join(
            os.getcwd(),
            'symlinks/data/glove/proc/glove_6B_300d_word_to_idx.json')
    else:
        # Glove + Visual
        model_const.embed2class.embed_dims = 350
        embed_dir = os.path.join(
            os.getcwd(),
            'symlinks/exp/cooccur/imagenet_genome_gt/' + \
            'training_no_fx_self_count_dim_50_single_embed/concat_with_glove_300')
        model_const.embed2class.embed_h5py = os.path.join(
            embed_dir,
            'visual_word_vecs.h5py')
        model_const.embed2class.embed_word_to_idx_json = os.path.join(
            embed_dir,
            'visual_word_vecs_idx.json')

    train.main(exp_const,data_const,model_const)


def exp_eval():
    #exp_name = 'training_no_fx_self_count_dim_50_single_embed_concat_with_glove_300_sgd_0.01'
    exp_name = 'glove_sgd_0.01'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.batch_size = 128
    exp_const.num_workers = 5

    data_const = Cifar100DatasetConstants()
    data_const.train = False

    model_const = Constants()
    model_const.model_num = 36000
    model_const.net = ResnetConstants()
    model_const.net.num_layers = 32
    model_const.net.num_classes = 100
    model_const.net.pretrained = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')
    model_const.embed2class = Embed2ClassConstants()
    model_const.embed2class_path = os.path.join(
        exp_const.model_dir,
        f'embed2class_{model_const.model_num}')
    model_const.embed2class.embed_dims = 300

    evaluation.main(exp_const,data_const,model_const)
    vis_conf_mat.main(exp_const)


def exp_vis_confmat():
    exp_name = 'training_no_fx_self_count_dim_50_single_embed_concat_with_glove_300_sgd_0.01'
    #exp_name = 'glove_sgd_0.01'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')

    vis_conf_mat.main(exp_const)


def exp_conf_vs_visual_sim():
    exp_name = 'training_no_fx_self_count_dim_50_single_embed_concat_with_glove_300_sgd_0.01'
    #exp_name = 'glove_sgd_0.01'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')

    data_const = Constants()
    visual_dir = exp_const.exp_dir
    data_const.visual_confmat_npy = os.path.join(visual_dir,'confmat.npy')
    data_const.visual_embed_npy = os.path.join(visual_dir,'embeddings.npy')
    data_const.labels_npy = os.path.join(visual_dir,'labels.npy')
    data_const.glove_confmat_npy = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100/glove_sgd_0.01/confmat.npy')
    data_const.glove_dim = 300

    conf_vs_visual_sim.main(exp_const,data_const)


if __name__=='__main__':
    list_exps(globals())
