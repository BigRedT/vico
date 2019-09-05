import os
import copy

from exp.experimenter import *
import utils.io as io
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from . import merge_cooccur
from .dataset import MultiSenseCooccurDatasetConstants
from .models.logbilinear import LogBilinearConstants
from . import train
from . import extract_embeddings, extract_embeddings_xformed
from . import concat_with_glove
from . import concat_random_with_glove
from . import synset_to_word_cooccur
from .vis import pca_tsne
from .vis import supervised_partitioning
from .vis import unsupervised_clustering
from data.glove.constants import GloveConstantsFactory
from data.visualgenome.constants import VisualGenomeConstants


parser.add_argument(
    '--embed_dim',
    type=int,
    default=100,
    help='Dimension of the embedding')
parser.add_argument(
    '--xform',
    type=str,
    default='linear',
    choices=['linear', 'select'],
    help='Type of transform in the multitask model')
parser.add_argument(
    '--model_num',
    type=int,
    default=-1,
    help='model num for finetuning or embedding extraction')
parser.add_argument(
    '--glove_dim',
    type=int,
    default=300,
    help='Dimension of GloVe embeddings to concatenate with ViCo')
parser.add_argument(
    '--syn',
    type=str_to_bool,
    default=True,
    help='Set to False to disable Synonyms during training')


def exp_synset_to_word_cooccur():
    paths = [
        ['imagenet','gt_obj_hyp_cooccur'],
        ['genome_attributes','gt_attr_attr_cooccur'],
        ['genome_attributes','gt_obj_attr_cooccur'],
        ['genome_attributes','gt_context_cooccur'],
    ]
    for dataset,exp_name in paths:
        out_base_dir = os.path.join(
            os.getcwd(),
            f'symlinks/exp/{dataset}')
        exp_const = ExpConstants(exp_name,out_base_dir)

        data_const = Constants()
        data_const.synset_cooccur_json = os.path.join(
            exp_const.exp_dir,
            'synset_cooccur.json')
        data_const.word_cooccur_json = os.path.join(
            exp_const.exp_dir,
            'word_cooccur.json')

        synset_to_word_cooccur.main(exp_const,data_const)


def exp_merge_cooccur():
    exp_name = 'cooccurrences'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.normalize = False

    data_const = Constants()
    cooccur_paths = {
        'syn': 'wordnet/syn_cooccur/word_cooccur.json',
        'attr_attr': 'genome_attributes/gt_attr_attr_cooccur/word_cooccur.json',
        'obj_attr': 'genome_attributes/gt_obj_attr_cooccur/word_cooccur.json',
        'obj_hyp': 'imagenet/gt_obj_hyp_cooccur/word_cooccur.json',
        'context': 'genome_attributes/gt_context_cooccur/word_cooccur.json',
    }
    symlink_exp = os.path.join(os.getcwd(),'symlinks/exp')
    data_const.cooccur_paths = {
        k: os.path.join(symlink_exp,v) for k,v in cooccur_paths.items()}
    data_const.merged_cooccur_csv = os.path.join(
        exp_const.exp_dir,
        'merged_cooccur.csv')
    merge_cooccur.main(exp_const,data_const)


def exp_train():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'embed_dim',
            'xform',
            'model_num',
            'syn'])

    exp_name = f'{args.xform}_{args.embed_dim}'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 100
    exp_const.model_save_step = 10000
    exp_const.batch_size = 1000
    exp_const.num_epochs = 10
    exp_const.lr = 0.01
    exp_const.momentum = 0.9    # used only when optimizer is set to 'SGD'
    exp_const.num_workers = 5
    # First train with Adam then finetune with Adagrad
    if args.model_num==-1:
        exp_const.optimizer = 'Adam'
    else:
        exp_const.optimizer = 'Adagrad'
    exp_const.weight_decay = 0
    exp_const.cooccur_weights = {
        'syn': 1,
        'attr_attr': 1,
        'obj_attr': 1,
        'obj_hyp': 1,
        'context': 1,
    }
    if args.syn==False:
        del exp_const.cooccur_weights['syn']

    exp_const.use_neg = True
    
    data_const = MultiSenseCooccurDatasetConstants()
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/cooccurrences/merged_cooccur.csv')
    data_const.use_self_count = True

    model_const = Constants()
    if args.model_num==-1:
        model_const.model_num = None
    else:
        model_const.model_num = args.model_num
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 93553
    model_const.net.embed_dims = args.embed_dim
    model_const.net.two_embedding_layers = False
    model_const.net.xform_type = args.xform
    model_const.net.xform_num_layers = None
    model_const.net.use_bias = True
    model_const.net.use_fx = False
    model_const.net.cooccur_types = [
        'syn',
        'attr_attr',
        'obj_attr',
        'obj_hyp',
        'context'
    ]
    if args.syn==False:
        model_const.net.cooccur_types = model_const.net.cooccur_types[1:]

    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    train.main(exp_const,data_const,model_const)


def exp_extract_embeddings():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'embed_dim',
            'xform',
            'model_num',
            'syn'])

    exp_name = f'{args.xform}_{args.embed_dim}'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.cooccur_types = [
        'syn',
        'attr_attr',
        'obj_attr',
        'obj_hyp',
        'context'
    ]
    if args.syn==False:
        exp_const.cooccur_types = exp_const.cooccur_types[1:]

    data_const = MultiSenseCooccurDatasetConstants()
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/cooccurrences/merged_cooccur.csv')

    model_const = Constants()
    model_const.model_num = args.model_num
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 93553
    model_const.net.embed_dims = args.embed_dim
    model_const.net.two_embedding_layers = False
    model_const.net.xform_type = args.xform
    model_const.net.xform_num_layers = None
    model_const.net.use_bias = True
    model_const.net.use_fx = False
    model_const.net.cooccur_types = copy.deepcopy(exp_const.cooccur_types)
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    extract_embeddings.main(exp_const,data_const,model_const)
    extract_embeddings_xformed.main(exp_const,data_const,model_const)


def exp_concat_with_glove():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'embed_dim',
            'xform',
            'glove_dim'])

    exp_name = f'concat_with_glove_{args.glove_dim}' # alt. xformed_
    out_base_dir = os.path.join(
        os.getcwd(),
        f'symlinks/exp/multi_sense_cooccur/{args.xform}_{args.embed_dim}')
    exp_const = ExpConstants(exp_name,out_base_dir)

    visual_embed_dir = exp_const.out_base_dir
    data_const = Constants()
    data_const.visual_word_to_idx = os.path.join(
        visual_embed_dir,
        'word_to_idx.json')
    data_const.visual_embeddings_npy = os.path.join(
        visual_embed_dir,
        'visual_embeddings.npy') # alt. _xformed.npy
    glove_const = GloveConstantsFactory.create(dim=str(args.glove_dim))
    data_const.glove_idx = glove_const.word_to_idx_json
    data_const.glove_h5py = glove_const.embeddings_h5py

    concat_with_glove.main(exp_const,data_const)


def exp_concat_random_with_glove():
    exp_name = 'concat_with_glove_100' # alt. xformed_
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/linear_100')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.random_dim = 100

    data_const = Constants()
    glove_const = GloveConstantsFactory.create(dim='100')
    data_const.glove_idx = glove_const.word_to_idx_json
    data_const.glove_h5py = glove_const.embeddings_h5py

    concat_random_with_glove.main(exp_const,data_const)


def exp_vis_pca_tsne():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'embed_dim',
            'xform',
            'glove_dim'])
    exp_name = f'{args.xform}_{args.embed_dim}'
    out_base_dir = os.path.join(
        os.getcwd(),
        f'symlinks/exp/multi_sense_cooccur')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis/tsne')
    exp_const.category_words_only = True

    data_const = Constants()
    embed_dir = os.path.join(
        exp_const.exp_dir,
        f'concat_with_glove_{args.glove_dim}')
    data_const.word_vecs_h5py = os.path.join(
        embed_dir,
        'visual_word_vecs.h5py')
    data_const.word_to_idx_json = os.path.join(
        embed_dir,
        'visual_word_vecs_idx.json')
    genome_const = VisualGenomeConstants()
    data_const.object_freqs_json = genome_const.object_freqs_json
    data_const.attribute_freqs_json = genome_const.attribute_freqs_json
    
    pca_tsne.main(exp_const,data_const)


class SimpleEmbedInfo():
    def __init__(self,word_to_idx_json,word_vecs_h5py):
        self.word_to_idx_json = word_to_idx_json # path to word_to_idx.json
        self.word_vecs_h5py = word_vecs_h5py # path to word

    def get_embedding(self,embeddings):
        return embeddings


class EmbedInfo():
        def __init__(self,exp_dir,random,to_extract,vico_dim,glove_dim=300):
            self.exp_dir = exp_dir
            self.random = random
            self.vico_dim = vico_dim
            self.glove_dim = glove_dim
            self.to_extract = to_extract
            
        @property
        def embed_dir(self):
            return os.path.join(
                self.exp_dir,
                f'concat_with_glove_{self.glove_dim}')
            
        @property
        def word_to_idx_json(self):
            return os.path.join(self.embed_dir,'visual_word_vecs_idx.json')
        
        @property
        def word_vecs_h5py(self):
            if self.random==True:
                h5py_path = os.path.join(
                    self.embed_dir,
                    'glove_random_word_vecs.h5py')
            else:
                h5py_path = os.path.join(
                    self.embed_dir,
                    'visual_word_vecs.h5py')
            return h5py_path

        def get_embedding(self,embeddings):
            if self.to_extract=='glove':
                embed = embeddings[:,:self.glove_dim]
            elif self.to_extract=='visual':
                embed = embeddings[:,self.glove_dim:]
            elif self.to_extract=='both':
                embed = embeddings
            else:
                assert(False),'extract type not implemented'
            
            return embed


def exp_unsupervised_clustering():
    exp_name = 'unsupervised_clustering'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/analysis')
    exp_const = ExpConstants(exp_name,out_base_dir)

    data_const = Constants()

    glove_vico_linear_100 = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/linear_100')

    """
    Update data_const.embed_info dictionary to control which embeddings are 
    evaluated. To evaluate your own embeddings create a class object with the 
    following 2 attributes:
    - `word_to_idx_json`: path to your word_to_idx.json file
    - `word_vecs_h5py`: path to your word_vecs.h5py file
    The class should also have a `get_embedding` function that accepts the
    read embeddings as argument and returns a modified version of it (eg. 
    reading only some of the embedding dimensions). 
     
    Class `EmbedInfo` is an example of such a class which dynamically creates 
    the 2 attributes and get_embedding function based on certain arguments.

    However the simplest case could look like `SimpleEmbedInfo` above.
    """
    data_const.embed_info = {
        'GloVe': EmbedInfo(
            glove_vico_linear_100,
            False,
            'glove', # Only glove component
            vico_dim=100,
            glove_dim=300), 
        'ViCo(linear,100)': EmbedInfo(
            glove_vico_linear_100,
            False,
            'visual', # Only visual component
            vico_dim=100,
            glove_dim=300), 
        'GloVe+ViCo(linear,100)': EmbedInfo(
            glove_vico_linear_100,
            False,
            'both', # Concatenated
            vico_dim=100,
            glove_dim=300), 
        # 'GloVe+ViCo(linear,100)[paper]': SimpleEmbedInfo(
        #     os.path.join(
        #         os.getcwd(),
        #         'symlinks/exp/multi_sense_cooccur/paper/linear_100/visual_word_vecs_idx.json'),
        #     os.path.join(
        #         os.getcwd(),
        #         'symlinks/exp/multi_sense_cooccur/paper/linear_100/visual_word_vecs.h5py'))
    }
    
    exp_const.fine = True
    unsupervised_clustering.main(exp_const,data_const)

    exp_const.fine = False
    unsupervised_clustering.main(exp_const,data_const)



def exp_supervised_partitioning():
    exp_name = 'supervised_partitioning'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/analysis')
    exp_const = ExpConstants(exp_name,out_base_dir)

    data_const = Constants()

    glove_vico_linear_100 = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/linear_100')

    """
    Update data_const.embed_info dictionary to control which embeddings are 
    evaluated. To evaluate your own embeddings create a class object with the 
    following 2 attributes:
    - `word_to_idx_json`: path to your word_to_idx.json file
    - `word_vecs_h5py`: path to your word_vecs.h5py file
    The class should also have a `get_embedding` function that accepts the
    read embeddings as argument and returns a modified version of it (eg. 
    reading only some of the embedding dimensions). 
     
    Class `EmbedInfo` is an example of such a class which dynamically creates 
    the 2 attributes and get_embedding function based on certain arguments.

    However the simplest case could look like `SimpleEmbedInfo` above.
    """
    data_const.embed_info = {
        'GloVe': EmbedInfo(
            glove_vico_linear_100,
            False,
            'glove', # Only glove component
            vico_dim=100,
            glove_dim=300), 
        'ViCo(linear,100)': EmbedInfo(
            glove_vico_linear_100,
            False,
            'visual', # Only visual component
            vico_dim=100,
            glove_dim=300), 
        'GloVe+ViCo(linear,100)': EmbedInfo(
            glove_vico_linear_100,
            False,
            'both', # Concatenated
            vico_dim=100,
            glove_dim=300),
        # 'GloVe+ViCo(linear,100)[paper]': SimpleEmbedInfo(
        #     os.path.join(
        #         os.getcwd(),
        #         'symlinks/exp/multi_sense_cooccur/paper/linear_100/visual_word_vecs_idx.json'),
        #     os.path.join(
        #         os.getcwd(),
        #         'symlinks/exp/multi_sense_cooccur/paper/linear_100/visual_word_vecs.h5py'))
    }
    
    exp_const.fine = True
    supervised_partitioning.main(exp_const,data_const)

    exp_const.fine = False
    supervised_partitioning.main(exp_const,data_const)


if __name__=='__main__':
    list_exps(globals())