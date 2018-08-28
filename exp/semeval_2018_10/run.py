import os
import h5py

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from exp.semeval_2018_10.dataset import SemEval201810DatasetConstants
from exp.semeval_2018_10.models.concat_mlp import ConcatMLPConstants
from exp.semeval_2018_10.models.concat_svm import ConcatSVMConstants
from data.glove.constants import GloveConstantsFactory
import exp.semeval_2018_10.train_concat_mlp as train_concat_mlp
import exp.semeval_2018_10.eval_concat_mlp as eval_concat_mlp
import exp.semeval_2018_10.train_concat_svm as train_concat_svm
import exp.semeval_2018_10.eval_concat_svm as eval_concat_svm

parser.add_argument(
    '--exp_name',
    type=str,
    help='Name of the experiment')
parser.add_argument(
    '--out_base_dir',
    default=None,
    type=str,
    help='Name of the output base directory. Uses default if set None.')
parser.add_argument(
    '--embed_linear_feat',
    default=True,
    type=str_to_bool,
    help='Set flag to use embedding linear features')
parser.add_argument(
    '--embed_quadratic_feat',
    default=True,
    type=str_to_bool,
    help='Set flag to use embedding quadratic features')
parser.add_argument(
    '--distance_linear_feat',
    default=True,
    type=str_to_bool,
    help='Set flag to use distance linear features')
parser.add_argument(
    '--distance_quadratic_feat',
    default=True,
    type=str_to_bool,
    help='Set flag to use distance quadratic features')
parser.add_argument(
    '--l2_weight',
    default=1e-3,
    type=float,
    help='Weight of the l2 regularization term for SVM')
parser.add_argument(
    '--lr',
    default=1e-2,
    type=float,
    help='Learning rate')
parser.add_argument(
    '--embed_dim',
    default=300,
    type=float,
    help='Embedding dimension')
parser.add_argument(
    '--embeddings_h5py',
    default=None,
    type=str,
    help='Path to embeddings h5py file')
parser.add_argument(
    '--word_to_idx_json',
    default=None,
    type=str,
    help='Path to word_to_idx.json file')
parser.add_argument(
    '--batch_size',
    default=2560,
    type=float,
    help='Embedding dimension')

def exp_train_concat_mlp():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[],
        optional_args=[])

    exp_name = 'trial'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/semeval_2018_10/concat_mlp')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_epochs = 10
    exp_const.batch_size = 128
    exp_const.lr = 1e-3

    data_const = SemEval201810DatasetConstants()
    glove_const = GloveConstantsFactory.create()
    data_const.embeddings_h5py = glove_const.embeddings_h5py
    data_const.word_to_idx_json = glove_const.word_to_idx_json
    
    model_const = Constants()
    model_const.concat_mlp = ConcatMLPConstants()
    model_const.concat_mlp.layer_units = []
    model_const.concat_mlp.drop_prob = 0.5
    model_const.concat_mlp.out_drop_prob = 0
    model_const.concat_mlp.use_bn = False

    train_concat_mlp.main(exp_const,data_const,model_const)


def exp_eval_concat_mlp():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['exp_name'],
        optional_args=['out_base_dir'])

    if args.out_base_dir is None:
        out_base_dir = os.path.join(
            os.getcwd(),
            'symlinks/exp/semeval_2018_10/concat_mlp')
    else:
        out_base_dir = args.out_base_dir

    exp_const = ExpConstants(
        exp_name=args.exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.batch_size = 128
    
    data_const = SemEval201810DatasetConstants()
    glove_const = GloveConstantsFactory.create()
    data_const.embeddings_h5py = glove_const.embeddings_h5py
    data_const.word_to_idx_json = glove_const.word_to_idx_json
    data_const.subset = 'test'
    
    model_const = Constants()
    model_const.concat_mlp = ConcatMLPConstants()
    model_const.concat_mlp.layer_units = []
    model_const.concat_mlp.drop_prob = 0.5
    model_const.concat_mlp.out_drop_prob = 0
    model_const.concat_mlp.use_bn = False

    eval_concat_mlp.main(exp_const,data_const,model_const)


def exp_train_concat_svm():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'lr',
            'l2_weight',
            'batch_size',
            'embed_linear_feat',
            'embed_quadratic_feat',
            'distance_linear_feat',
            'distance_quadratic_feat'
        ],
        optional_args=[
            'exp_name',
            'out_base_dir',
            'embeddings_h5py',
            'word_to_idx_json'])

    if args.exp_name is None:
        exp_name = 'trial'
    else:
        exp_name = args.exp_name

    if args.out_base_dir is None:
        out_base_dir = os.path.join(
            os.getcwd(),
            'symlinks/exp/semeval_2018_10/concat_svm')
    else:
        out_base_dir = args.out_base_dir

    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_epochs = 20
    exp_const.batch_size = 2560
    exp_const.lr = args.lr

    data_const = SemEval201810DatasetConstants()
    if args.embeddings_h5py is None:
        glove_const = GloveConstantsFactory.create()
        data_const.embeddings_h5py = glove_const.embeddings_h5py
        data_const.word_to_idx_json = glove_const.word_to_idx_json
    else:
        data_const.embeddings_h5py = args.embeddings_h5py
        data_const.word_to_idx_json = args.word_to_idx_json
    
    embed_dim = h5py.File(
        data_const.embeddings_h5py,
        'r')['embeddings'].shape[1]

    model_const = Constants()
    model_const.concat_svm = ConcatSVMConstants()
    model_const.concat_svm.l2_weight = 0.01*args.l2_weight
    model_const.concat_svm.embedding_dim = embed_dim
    model_const.concat_svm.layer_units = []
    model_const.concat_svm.use_embedding_linear_feats = args.embed_linear_feat
    model_const.concat_svm.use_embedding_quadratic_feats = \
        args.embed_quadratic_feat
    model_const.concat_svm.use_distance_linear_feats = args.distance_linear_feat
    model_const.concat_svm.use_distance_quadratic_feats = \
        args.distance_quadratic_feat

    train_concat_svm.main(exp_const,data_const,model_const)


def exp_eval_concat_svm():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'batch_size',
            'embed_linear_feat',
            'embed_quadratic_feat',
            'distance_linear_feat',
            'distance_quadratic_feat'
        ],
        optional_args=[
            'exp_name',
            'out_base_dir',
            'embeddings_h5py',
            'word_to_idx_json'])

    if args.exp_name is None:
        exp_name = 'trial'
    else:
        exp_name = args.exp_name

    if args.out_base_dir is None:
        out_base_dir = os.path.join(
            os.getcwd(),
            'symlinks/exp/semeval_2018_10/concat_svm')
    else:
        out_base_dir = args.out_base_dir

    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.batch_size = 2560
    
    data_const = SemEval201810DatasetConstants()
    data_const.subset = 'test'
    if args.embeddings_h5py is None:
        glove_const = GloveConstantsFactory.create()
        data_const.embeddings_h5py = glove_const.embeddings_h5py
        data_const.word_to_idx_json = glove_const.word_to_idx_json
    else:
        data_const.embeddings_h5py = args.embeddings_h5py
        data_const.word_to_idx_json = args.word_to_idx_json
    
    embed_dim = h5py.File(
        data_const.embeddings_h5py,
        'r')['embeddings'].shape[1]

    model_const = Constants()
    model_const.concat_svm = ConcatSVMConstants()
    model_const.concat_svm.embedding_dim = embed_dim
    model_const.concat_svm.layer_units = []
    model_const.concat_svm.use_embedding_linear_feats = args.embed_linear_feat
    model_const.concat_svm.use_embedding_quadratic_feats = \
        args.embed_quadratic_feat
    model_const.concat_svm.use_distance_linear_feats = args.distance_linear_feat
    model_const.concat_svm.use_distance_quadratic_feats = \
        args.distance_quadratic_feat

    eval_concat_svm.main(exp_const,data_const,model_const)


if __name__=='__main__':
    list_exps(globals())