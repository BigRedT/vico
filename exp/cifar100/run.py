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
from .vis import conf_vs_visual_sim, conf_as_fun_of_sims, class_vs_sim
from .vis import acc_vs_num_classes, acc_with_std_vs_num_classes


parser.add_argument(
    '--held_classes',
    type=int,
    help='number of held out classes')
parser.add_argument(
    '--embed_type',
    type=str,
    choices=['random','glove','glove_random',
        'glove_vico_linear','glove_vico_select'],
    help='embedding types')
parser.add_argument(
    '--vico_dim',
    type=int,
    help='dimension of vico/random embeddings')


def exp_train():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'held_classes',
            'embed_type',
            'vico_dim'],
        optional_args=[])
    exp_name = \
        args.embed_type + '_' + str(args.vico_dim) + '_' + \
        'held_classes_' + str(args.held_classes)
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100/zero_shot4')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 200
    exp_const.model_save_step = 1000
    exp_const.val_step = 1000
    exp_const.batch_size = 128
    exp_const.num_epochs = 50 #100
    # Note on LR
    # - 0.01 with adam; finetune with 0.01 sgd for simple resnet
    # - 0.01 with sgd for 32K iters and then 0.001 sgd for embedding resnet
    exp_const.lr = 0.01 
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.optimizer = 'Adam'
    exp_const.feedforward = False
    exp_const.subset = {
        'training': 'train',
        'test': 'test'
    }

    data_const = Cifar100DatasetConstants()
    data_const.num_held_out_classes = args.held_classes

    model_const = Constants()
    model_const.model_num = None #32000 #, 48000
    model_const.net = ResnetConstants()
    model_const.net.num_layers = 32
    model_const.net.num_classes = 100
    model_const.net.pretrained = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')
    model_const.embed2class = Embed2ClassConstants()
    model_const.embed2class.linear = True
    model_const.embed2class_path = os.path.join(
        exp_const.model_dir,
        f'embed2class_{model_const.model_num}')
    model_const.embed2class.glove_dim = 100

    # Dimensions
    if args.embed_type=='glove':
        model_const.embed2class.embed_dims = model_const.embed2class.glove_dim
    else:
        model_const.embed2class.embed_dims = \
            model_const.embed2class.glove_dim + \
            args.vico_dim
        
        if args.embed_type=='random':
            # for 'random' glove dimensions are zeroed out by setting no_glove
            model_const.embed2class.no_glove = True

    print('Full embedding dimension: ',model_const.embed2class.embed_dims)

    # embeddings path
    if args.embed_type=='glove':
        model_const.embed2class.embed_h5py = os.path.join(
            os.getcwd(),
            'symlinks/data/glove/proc/glove_6B_100d.h5py')
        model_const.embed2class.embed_word_to_idx_json = os.path.join(
            os.getcwd(),
            'symlinks/data/glove/proc/glove_6B_100d_word_to_idx.json')
    else:
        if args.embed_type in ['random','glove_random']:
            embed_dir = os.path.join(
                os.getcwd(),
                'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/' + \
                f'effect_of_xforms/dim_{args.vico_dim}_neg_bias_linear/' + \
                'concat_with_glove_100')
            model_const.embed2class.embed_h5py = os.path.join(
                embed_dir,
                'glove_random_word_vecs.h5py')
        elif args.embed_type=='glove_vico_linear':
            embed_dir = os.path.join(
                os.getcwd(),
                'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/' + \
                f'effect_of_xforms/dim_{args.vico_dim}_neg_bias_linear/' + \
                'concat_with_glove_100')
            model_const.embed2class.embed_h5py = os.path.join(
                embed_dir,
                'visual_word_vecs.h5py')
        elif args.embed_type=='glove_vico_select':
            embed_dir = os.path.join(
                os.getcwd(),
                'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/' + \
                f'effect_of_xforms/dim_{args.vico_dim}_neg_bias_select/' + \
                'concat_with_glove_100')
            model_const.embed2class.embed_h5py = os.path.join(
                embed_dir,
                'visual_word_vecs.h5py')
        else:
            err_msg = f'{args.embed_type} not implemented'
            assert(False), err_msg

        model_const.embed2class.embed_word_to_idx_json = os.path.join(
            embed_dir,
            'visual_word_vecs_idx.json')

    train.main(exp_const,data_const,model_const)


def exp_acc_vs_num_train_classes():
    exp_name = 'agg_results'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100/zero_shot4')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.prefix = {
        'random(100)': 'random_100_held_classes_',
        'GloVe': 'glove_100_held_classes_', 
        'GloVe+random(100)': 'glove_random_100_held_classes_',
        'GloVe+random(200)': 'glove_random_200_held_classes_',
        'GloVe+ViCo(linear,100)': 'glove_vico_linear_100_held_classes_',
        'GloVe+ViCo(linear,200)': 'glove_vico_linear_200_held_classes_',
        'GloVe+ViCo(select,200)': 'glove_vico_select_200_held_classes_'}
    exp_const.held_out_classes = [20,40,60,80]

    acc_vs_num_classes.main(exp_const)


def exp_acc_with_std_vs_num_train_classes():
    exp_name = 'agg_results'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.runs_prefix = os.path.join(out_base_dir,'zero_shot')
    exp_const.runs = []
    for i in range(4):
        exp_const.runs.append(exp_const.runs_prefix + str(i+1))
    exp_const.prefix = {
        'random(100)': 'random_100_held_classes_',
        'GloVe': 'glove_100_held_classes_', 
        'GloVe+random(100)': 'glove_random_100_held_classes_',
        'GloVe+random(200)': 'glove_random_200_held_classes_',
        'GloVe+ViCo(linear,100)': 'glove_vico_linear_100_held_classes_',
        'GloVe+ViCo(linear,200)': 'glove_vico_linear_200_held_classes_',
        'GloVe+ViCo(select,200)': 'glove_vico_select_200_held_classes_'}
    exp_const.held_out_classes = [20,40,60,80]

    acc_with_std_vs_num_classes.main(exp_const)


def exp_eval():
    # exp_name = 'feedforward_sgd_weight_decay_data_aug'
    # exp_name = 'training_no_fx_self_count_dim_50_single_embed_concat_with_glove_300_sgd_0.01_linear'
    exp_name = 'glove_sgd_0.01_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.batch_size = 128
    exp_const.num_workers = 5
    exp_const.feedforward = False

    data_const = Cifar100DatasetConstants()
    data_const.train = False

    model_const = Constants()
    model_const.model_num = 34000
    model_const.net = ResnetConstants()
    model_const.net.num_layers = 32
    model_const.net.num_classes = 100
    model_const.net.pretrained = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')
    model_const.embed2class = Embed2ClassConstants()
    model_const.embed2class.linear = True
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
    exp_name = 'training_no_fx_self_count_dim_50_single_embed_concat_with_glove_300_sgd_0.01_linear'
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
        'symlinks/exp/cifar100/glove_sgd_0.01_linear/confmat.npy')
    data_const.glove_dim = 300

    conf_vs_visual_sim.main(exp_const,data_const)


def exp_conf_as_fun_of_sims():
    exp_name = 'training_no_fx_self_count_dim_50_single_embed_concat_with_glove_300_sgd_0.01_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.cosine = False

    data_const = Constants()
    visual_dir = exp_const.exp_dir
    data_const.visual_confmat_npy = os.path.join(visual_dir,'confmat.npy')
    data_const.visual_embed_npy = os.path.join(visual_dir,'embeddings.npy')
    data_const.labels_npy = os.path.join(visual_dir,'labels.npy')
    data_const.glove_confmat_npy = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100/glove_sgd_0.01/confmat.npy')
    data_const.glove_dim = 300

    conf_as_fun_of_sims.main(exp_const,data_const)


def exp_class_vs_sim():
    exp_name = 'training_no_fx_self_count_dim_50_single_embed_concat_with_glove_300_sgd_0.01_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')

    data_const = Constants()
    visual_dir = exp_const.exp_dir
    data_const.visual_embed_npy = os.path.join(visual_dir,'embeddings.npy')
    data_const.labels_npy = os.path.join(visual_dir,'labels.npy')
    data_const.class_confmat_npy = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100/feedforward_sgd_weight_decay_data_aug/confmat.npy')
    data_const.glove_dim = 300

    class_vs_sim.main(exp_const,data_const)


if __name__=='__main__':
    list_exps(globals())
