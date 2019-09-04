import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from .models.resnet import ResnetConstants
from .models.embed_to_classifier import Embed2ClassConstants
from .dataset import Cifar100DatasetConstants
from . import train
from . import agg_results


parser.add_argument(
    '--held_classes',
    type=int,
    help='number of held out classes')
parser.add_argument(
    '--embed_type',
    type=str,
    choices=['glove','vico_linear','glove_vico_linear','glove_vico_select'],
    help='embedding types')
parser.add_argument(
    '--vico_dim',
    type=int,
    help='dimension of vico embeddings')
parser.add_argument(
    '--glove_dim',
    type=int,
    default=300,
    help='Dimension of GloVe embeddings to concatenate with ViCo')
parser.add_argument(
    '--run',
    type=int,
    default=0,
    help='Run number')


def exp_train():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'held_classes',
            'embed_type',
            'glove_dim',
            'vico_dim',
            'run'],
        optional_args=[])
    exp_name = \
        args.embed_type + '_' + \
        str(args.glove_dim) + '_' + \
        str(args.vico_dim) + '_' + \
        'held_classes_' + str(args.held_classes)
    out_base_dir = os.path.join(
        os.getcwd(),
        f'symlinks/exp/cifar100/zero_shot_{args.run}')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 200
    exp_const.model_save_step = 1000
    exp_const.val_step = 1000
    exp_const.batch_size = 128
    exp_const.num_epochs = 50 #100
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
    model_const.model_num = None
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
    model_const.embed2class.glove_dim = args.glove_dim

    # Dimensions
    if args.embed_type=='glove':
        model_const.embed2class.embed_dims = args.glove_dim
        model_const.embed2class.embed_h5py = os.path.join(
            os.getcwd(),
            f'symlinks/data/glove/proc/glove_6B_{args.glove_dim}d.h5py')
        model_const.embed2class.embed_word_to_idx_json = os.path.join(
            os.getcwd(),
            f'symlinks/data/glove/proc/glove_6B_{args.glove_dim}d_word_to_idx.json')
    elif args.embed_type=='glove_vico_linear':
        model_const.embed2class.embed_dims = args.glove_dim + args.vico_dim
        embed_dir = os.path.join(
            os.getcwd(),
            'symlinks/exp/multi_sense_cooccur/' + \
            f'linear_100/concat_with_glove_{args.glove_dim}')
        model_const.embed2class.embed_h5py = os.path.join(
            embed_dir,
            'visual_word_vecs.h5py')
        model_const.embed2class.embed_word_to_idx_json = os.path.join(
            embed_dir,
            'visual_word_vecs_idx.json')
    elif args.embed_type=='vico_linear':
        model_const.embed2class.no_glove = True # Zero out the glove component
        model_const.embed2class.embed_dims = args.glove_dim + args.vico_dim
        embed_dir = os.path.join(
            os.getcwd(),
            'symlinks/exp/multi_sense_cooccur/' + \
            f'linear_100/concat_with_glove_{args.glove_dim}')
        model_const.embed2class.embed_h5py = os.path.join(
            embed_dir,
            'visual_word_vecs.h5py')
        model_const.embed2class.embed_word_to_idx_json = os.path.join(
            embed_dir,
            'visual_word_vecs_idx.json')
    elif args.embed_type=='glove_vico_select':
        model_const.embed2class.embed_dims = args.glove_dim + args.vico_dim
        embed_dir = os.path.join(
            os.getcwd(),
            'symlinks/exp/multi_sense_cooccur/' + \
            f'select_200/concat_with_glove_{args.glove_dim}')
        model_const.embed2class.embed_h5py = os.path.join(
            embed_dir,
            'visual_word_vecs.h5py')
        model_const.embed2class.embed_word_to_idx_json = os.path.join(
            embed_dir,
            'visual_word_vecs_idx.json')
    else:
        err_str = f'{args.embed_type} is currently not implemented in the runner'
        assert(False), err_str

    train.main(exp_const,data_const,model_const)


def exp_agg_results():
    exp_name = 'agg_results'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/cifar100')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.runs_prefix = os.path.join(out_base_dir,'zero_shot_')
    exp_const.runs = [0,1,2,3]
    exp_const.prefix = {
        'GloVe': 'glove_100_0_held_classes_', 
        #'ViCo(linear,100)': 'vico_linear_100_100_held_classes_',
        'GloVe+ViCo(linear,100)': 'glove_vico_linear_100_100_held_classes_',
        #'GloVe+ViCo(select,200)': 'glove_vico_select_100_200_held_classes_'
    }
    
    held_out_classes_list = [20]
    for held_out_classes in held_out_classes_list:
        exp_const.held_out_classes = held_out_classes
        agg_results.main(exp_const)


if __name__=='__main__':
    list_exps(globals())
