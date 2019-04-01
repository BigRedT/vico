import os

from exp.experimenter import *
import utils.io as io
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from . import merge_cooccur
from .dataset import MultiSenseCooccurDatasetConstants
from .models.logbilinear import LogBilinearConstants
from . import train
from . import train_w_neg
from . import extract_embeddings, extract_embeddings_xformed
from . import find_nn
from . import concat_with_glove
from . import concat_random_with_glove
from data.glove.constants import GloveConstantsFactory
from . import synset_to_word_cooccur
from .vis import bias_vs_self_count #as bias_vs_self_count
from .vis import pca_tsne
from .vis import supervised_clustering
from .vis import unsupervised_clustering
from .vis import glove_vs_vico_sim
from data.visualgenome.constants import VisualGenomeConstants


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
    exp_name = 'imagenet_genome_gt'
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
        'merged_cooccur_self.csv') # 'merged_cooccur.csv'
    merge_cooccur.main(exp_const,data_const)


def exp_merge_word_cooccur():
    exp_name = 'imagenet_genome_gt_word'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.normalize = False

    data_const = Constants()
    cooccur_paths = {
        #'syn': 'wordnet/syn_cooccur_self/word_cooccur.json',
        'attr_attr': 'genome_attributes/gt_attr_attr_word_cooccur_self/word_cooccur.json',
        'obj_attr': 'genome_attributes/gt_obj_attr_word_cooccur_self/word_cooccur.json',
        'obj_hyp': 'imagenet/gt_obj_hyp_word_cooccur_self/word_cooccur.json',
        'context': 'genome_attributes/gt_context_word_cooccur_self/word_cooccur.json',
    }
    symlink_exp = os.path.join(os.getcwd(),'symlinks/exp')
    data_const.cooccur_paths = {
        k: os.path.join(symlink_exp,v) for k,v in cooccur_paths.items()}
    data_const.merged_cooccur_csv = os.path.join(
        exp_const.exp_dir,
        'merged_cooccur_self.csv') # 'merged_cooccur.csv'
    merge_cooccur.main(exp_const,data_const)


def exp_train():
    exp_name = 'dim_100_neg_bias_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 10
    exp_const.model_save_step = 10000
    exp_const.batch_size = 1000
    exp_const.num_epochs = 10
    exp_const.lr = 0.01
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.optimizer = 'Adam' # First train with Adam then swith to Adagrad
    exp_const.weight_decay = 0*1e-4
    exp_const.cooccur_weights = {
        # 'syn': 0,
        'attr_attr': 1,
        'obj_attr': 1,
        'obj_hyp': 1,
        'context': 1,
    }
    exp_const.use_neg = True
    
    data_const = MultiSenseCooccurDatasetConstants()
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt/merged_cooccur_self.csv')
    data_const.use_self_count = True

    model_const = Constants()
    model_const.model_num = None # 50000
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 93553 #93553 for w synsets, 33444 wo
    model_const.net.embed_dims = 100
    model_const.net.two_embedding_layers = False
    model_const.net.xform_type = 'linear'
    model_const.net.xform_num_layers = None
    model_const.net.use_bias = True
    model_const.net.use_fx = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    #train.main(exp_const,data_const,model_const)
    train_w_neg.main(exp_const,data_const,model_const)


# Difference between exp_tran and exp_train_word: 
# - append "word" to paths
# - update num_words
def exp_train_word():
    exp_name = 'dim_100_neg_bias_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt_word/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.log_step = 10
    exp_const.model_save_step = 10000
    exp_const.batch_size = 1000
    exp_const.num_epochs = 10
    exp_const.lr = 0.01
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.optimizer = 'Adagrad'
    exp_const.weight_decay = 0*1e-4
    exp_const.cooccur_weights = {
        # 'syn': 0,
        'attr_attr': 1,
        'obj_attr': 1,
        'obj_hyp': 0,
        'context': 1,
    }
    exp_const.use_neg = True
    
    data_const = MultiSenseCooccurDatasetConstants()
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt_word/merged_cooccur_self.csv')
    data_const.use_self_count = True

    model_const = Constants()
    model_const.model_num = 30000 # 50000
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 33444 #93553 for w synsets, 33444 wo
    model_const.net.embed_dims = 100
    model_const.net.two_embedding_layers = False
    model_const.net.xform_type = 'linear'
    model_const.net.xform_num_layers = None
    model_const.net.use_bias = True
    model_const.net.use_fx = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    #train.main(exp_const,data_const,model_const)
    train_w_neg.main(exp_const,data_const,model_const)


def exp_extract_embeddings():
    exp_name = 'dim_100_neg_bias_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt_word/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.cooccur_types = [
        #'syn',
        'attr_attr',
        'obj_attr',
        'obj_hyp',
        'context']

    data_const = MultiSenseCooccurDatasetConstants()
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt_word/merged_cooccur_self.csv')

    model_const = Constants()
    model_const.model_num = 60000 #110000
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 33444 #93553
    model_const.net.embed_dims = 100
    model_const.net.two_embedding_layers = False
    model_const.net.xform_type = 'linear'
    model_const.net.xform_num_layers = None
    model_const.net.use_bias = True
    model_const.net.use_fx = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    extract_embeddings.main(exp_const,data_const,model_const)
    #extract_embeddings_xformed.main(exp_const,data_const,model_const)


def exp_find_nn():
    exp_name = 'dim_50_neg_bias_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_nbrs = 20
    exp_const.use_cosine = True
    exp_const.thresh = [
        #10000,
        100000,
        100000,
        100000,
        100000,
    ]
    exp_const.cooccur_types = [
        #'syn',
        'attr_attr',
        'obj_attr',
        'obj_hyp',
        'context'
    ]

    data_const = Constants()
    data_const.embeddings_npy = os.path.join(
        exp_const.exp_dir,
        'visual_embeddings.npy')
    data_const.word_to_idx_json = os.path.join(
        exp_const.exp_dir,
        'word_to_idx.json')
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt/merged_cooccur_self.csv')

    model_const = Constants()
    model_const.model_num = 120000
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 93553
    model_const.net.embed_dims = 50
    model_const.net.two_embedding_layers = False
    model_const.net.xform_type = 'linear'
    model_const.net.xform_num_layers = None
    model_const.net.use_bias = True
    model_const.net.use_fx = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')

    find_nn.main(exp_const,data_const,model_const)


def exp_concat_with_glove():
    exp_name = 'concat_with_glove_300' # alt. xformed_
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt_word/' + \
        'effect_of_xforms/dim_100_neg_bias_linear')
    exp_const = ExpConstants(exp_name,out_base_dir)

    visual_embed_dir = exp_const.out_base_dir
    data_const = Constants()
    data_const.visual_word_to_idx = os.path.join(
        visual_embed_dir,
        'word_to_idx.json')
    data_const.visual_embeddings_npy = os.path.join(
        visual_embed_dir,
        'visual_embeddings.npy') # alt. _xformed.npy
    glove_const = GloveConstantsFactory.create(dim='300')
    data_const.glove_idx = glove_const.word_to_idx_json
    data_const.glove_h5py = glove_const.embeddings_h5py

    concat_with_glove.main(exp_const,data_const)


def exp_concat_visw2v_with_glove():
    exp_name = 'concat_with_glove_300' # alt. xformed_
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/vis_w2v/visual_word2vec_wiki')
    exp_const = ExpConstants(exp_name,out_base_dir)

    visual_embed_dir = exp_const.out_base_dir
    data_const = Constants()
    data_const.visual_word_to_idx = os.path.join(
        visual_embed_dir,
        'word_to_idx.json')
    data_const.visual_embeddings_npy = os.path.join(
        visual_embed_dir,
        'visual_embeddings.npy') # alt. _xformed.npy
    glove_const = GloveConstantsFactory.create(dim='300')
    data_const.glove_idx = glove_const.word_to_idx_json
    data_const.glove_h5py = glove_const.embeddings_h5py

    concat_with_glove.main(exp_const,data_const)



def exp_concat_random_with_glove():
    exp_name = 'concat_with_glove_100' # alt. xformed_
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/' + \
        'effect_of_xforms/dim_100_neg_bias_linear')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.random_dim = 100

    data_const = Constants()
    glove_const = GloveConstantsFactory.create(dim='100')
    data_const.glove_idx = glove_const.word_to_idx_json
    data_const.glove_h5py = glove_const.embeddings_h5py

    concat_random_with_glove.main(exp_const,data_const)


def exp_vis_bias_vs_self_count():
    exp_name = 'dim_50_neg_bias_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')

    data_const = MultiSenseCooccurDatasetConstants()
    data_const.cooccur_csv = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/' + \
        'imagenet_genome_gt/merged_cooccur_self.csv')

    model_const = Constants()
    model_const.model_num = 120000
    model_const.net = LogBilinearConstants()
    model_const.net.num_words = 93553
    model_const.net.embed_dims = 50
    model_const.net.two_embedding_layers = False
    model_const.net.xform_type = 'linear'
    model_const.net.xform_num_layers = None
    model_const.net.use_bias = True
    model_const.net.use_fx = False
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')
    
    bias_vs_self_count.main(exp_const,data_const,model_const)


def exp_vis_pca_tsne():
    xformed = False
    exp_name = 'dim_100_neg_bias_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    if xformed==True:
        exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis/tsne_xformed')
    else:
        exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis/tsne')
    exp_const.category_words_only = True
    exp_const.xformed = xformed
    exp_const.cooccur_types = [
        'attr_attr',
        'obj_attr',
        'obj_hyp',
        'context'
    ]
    exp_const.cooccur_dim = 50 # after xform

    data_const = Constants()
    if xformed==True:
        embed_dir = os.path.join(
            exp_const.exp_dir,
            'xformed_concat_with_glove_300')
    else:
        embed_dir = os.path.join(
            exp_const.exp_dir,
            'concat_with_glove_300')
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


def exp_supervised_clustering_bkp():
    exp_name = 'dim_100_neg_bias_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis/clustering_w_select')
    exp_const.fine = True

    data_const = Constants()
    
    embed_dir = os.path.join(
        exp_const.exp_dir,
        'concat_with_glove_300')
    data_const.word_vecs_h5py = os.path.join(
        embed_dir,
        'visual_word_vecs.h5py')
    data_const.word_to_idx_json = os.path.join(
        embed_dir,
        'visual_word_vecs_idx.json')
    
    xformed_embed_dir = os.path.join(
        exp_const.exp_dir,
        'xformed_concat_with_glove_300')
    data_const.xformed_word_vecs_h5py = os.path.join(
        xformed_embed_dir,
        'visual_word_vecs.h5py')

    data_const.select_word_vecs_h5py = os.path.join(
        out_base_dir,
        'dim_200_neg_bias_select/concat_with_glove_300/' + \
        'visual_word_vecs.h5py')
    
    supervised_clustering.main(exp_const,data_const)


def exp_unsupervised_clustering_bkp():
    exp_name = 'dim_100_neg_bias_linear'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis/clustering_w_select')
    exp_const.fine = False

    data_const = Constants()
    
    embed_dir = os.path.join(
        exp_const.exp_dir,
        'concat_with_glove_300')
    data_const.word_vecs_h5py = os.path.join(
        embed_dir,
        'visual_word_vecs.h5py')
    data_const.word_to_idx_json = os.path.join(
        embed_dir,
        'visual_word_vecs_idx.json')
    
    xformed_embed_dir = os.path.join(
        exp_const.exp_dir,
        'xformed_concat_with_glove_300')
    data_const.xformed_word_vecs_h5py = os.path.join(
        xformed_embed_dir,
        'visual_word_vecs.h5py')

    data_const.select_word_vecs_h5py = os.path.join(
        out_base_dir,
        'dim_200_neg_bias_select/concat_with_glove_300/' + \
        'visual_word_vecs.h5py')
    
    unsupervised_clustering.main(exp_const,data_const)


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


class SelectEmbedInfo(EmbedInfo):
    def __init__(
            self,
            exp_dir,
            random,
            to_extract,
            vico_dim,
            cooccur_type,
            glove_dim=300):
        super(SelectEmbedInfo,self).__init__(
            exp_dir,
            random,
            to_extract,
            vico_dim,
            glove_dim)
        self.cooccur_type = cooccur_type
        
    def get_embedding(self,embeddings):
        embed = super(SelectEmbedInfo,self).get_embedding(embeddings)
        if self.to_extract!='visual':
            return embed
        
        if self.cooccur_type=='attr_attr':
            return embed[:,:50]
        elif self.cooccur_type=='obj_attr':
            return embed[:,50:100]
        elif self.cooccur_type=='obj_hyp':
            return embed[:,100:150]
        elif self.cooccur_type=='context':
            return embed[:,150:200]
        else:
            assert(False),'cooccur_type not implemented'

        

def exp_unsupervised_clustering():
    exp_name = 'analysis'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis/clustering')
    exp_const.fine = False

    data_const = Constants()

    glove_vico_linear_50 = os.path.join(out_base_dir,'dim_50_neg_bias_linear')
    glove_vico_linear_100 = os.path.join(out_base_dir,'dim_100_neg_bias_linear')
    glove_vico_linear_200 = os.path.join(out_base_dir,'dim_200_neg_bias_linear')
    glove_vico_select_200 = os.path.join(out_base_dir,'dim_200_neg_bias_select')
    no_wn_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt_word/effect_of_xforms')
    glove_vico_wo_wordnet_linear_100 = os.path.join(
        no_wn_dir,
        'dim_100_neg_bias_linear')
    glove_vis_w2v_200 = os.path.join(
        os.getcwd(),
        'symlinks/exp/vis_w2v/visual_word2vec_wiki')
    data_const.embed_info = {
        # 'random(300)': EmbedInfo(glove_vico_linear_100,True,'visual',300),
        # 'random(100)': EmbedInfo(glove_vico_linear_100,True,'visual',100),
        # 'GloVe': EmbedInfo(glove_vico_linear_100,True,'glove',100),
        # 'GloVe+random(100)': EmbedInfo(glove_vico_linear_100,True,'both',100),
        # 'GloVe+random(200)': EmbedInfo(glove_vico_linear_200,True,'both',200),
        'ViCo(linear,50)': EmbedInfo(glove_vico_linear_50,False,'visual',50),
        'ViCo(linear,100)': EmbedInfo(glove_vico_linear_100,False,'visual',100),
        'ViCo(linear,200)': EmbedInfo(glove_vico_linear_200,False,'visual',200),
        'ViCo(select,200)': EmbedInfo(glove_vico_select_200,False,'visual',200),
        # 'GloVe+ViCo(linear,100, w/o WordNet)': EmbedInfo(
        #     glove_vico_wo_wordnet_linear_100,False,'both',100),
        # 'GloVe+ViCo(linear,100)': EmbedInfo(glove_vico_linear_100,False,'both',100),
        # 'GloVe+ViCo(linear,200)': EmbedInfo(glove_vico_linear_200,False,'both',200),
        # 'GloVe+ViCo(select,200)': EmbedInfo(glove_vico_select_200,False,'both',200),
        #'GloVe+ViCo(linear,50)': EmbedInfo(glove_vico_linear_50,False,'both',50),
        'vis-w2v': EmbedInfo(glove_vis_w2v_200,False,'visual',200),
        'GloVe+vis-w2v': EmbedInfo(glove_vis_w2v_200,False,'both',200)
    }
    
    unsupervised_clustering.main(exp_const,data_const)



def exp_supervised_clustering():
    exp_name = 'analysis'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis/clustering')
    exp_const.fine = False

    data_const = Constants()

    glove_vico_linear_50 = os.path.join(out_base_dir,'dim_50_neg_bias_linear')
    glove_vico_linear_100 = os.path.join(out_base_dir,'dim_100_neg_bias_linear')
    glove_vico_linear_200 = os.path.join(out_base_dir,'dim_200_neg_bias_linear')
    glove_vico_select_200 = os.path.join(out_base_dir,'dim_200_neg_bias_select')
    no_wn_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt_word/effect_of_xforms')
    glove_vico_wo_wordnet_linear_100 = os.path.join(
        no_wn_dir,
        'dim_100_neg_bias_linear')
    glove_vis_w2v_200 = os.path.join(
        os.getcwd(),
        'symlinks/exp/vis_w2v/visual_word2vec_wiki')
    data_const.embed_info = {
        # 'random(300)': EmbedInfo(glove_vico_linear_100,True,'visual',300),
        # 'random(100)': EmbedInfo(glove_vico_linear_100,True,'visual',100),
        # 'GloVe': EmbedInfo(glove_vico_linear_100,True,'glove',100),
        # 'GloVe+random(100)': EmbedInfo(glove_vico_linear_100,True,'both',100),
        # 'GloVe+random(200)': EmbedInfo(glove_vico_linear_200,True,'both',200),
        'ViCo(linear,50)': EmbedInfo(glove_vico_linear_50,False,'visual',50),
        'ViCo(linear,100)': EmbedInfo(glove_vico_linear_100,False,'visual',100),
        'ViCo(linear,200)': EmbedInfo(glove_vico_linear_200,False,'visual',200),
        'ViCo(select,200)': EmbedInfo(glove_vico_select_200,False,'visual',200),
        # 'GloVe+ViCo(linear,100, w/o WordNet)': EmbedInfo(
        #     glove_vico_wo_wordnet_linear_100,False,'both',100),
        # 'GloVe+ViCo(linear,100)': EmbedInfo(glove_vico_linear_100,False,'both',100),
        # 'GloVe+ViCo(linear,200)': EmbedInfo(glove_vico_linear_200,False,'both',200),
        # 'GloVe+ViCo(select,200)': EmbedInfo(glove_vico_select_200,False,'both',200),
        #'GloVe+ViCo(linear,50)': EmbedInfo(glove_vico_linear_50,False,'both',50),
        'vis-w2v': EmbedInfo(glove_vis_w2v_200,False,'visual',200),
        'GloVe+vis-w2v': EmbedInfo(glove_vis_w2v_200,False,'both',200)
    }
    
    supervised_clustering.main(exp_const,data_const)


def exp_glove_vs_vico_sim():
    exp_name = 'analysis'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis/glove_vs_vico_sim')
    exp_const.fine = True

    data_const = Constants()
    glove_vico_linear_100 = os.path.join(out_base_dir,'dim_100_neg_bias_linear')
    glove_vico_select_200 = os.path.join(out_base_dir,'dim_200_neg_bias_select')
    data_const.embed_info = {
        'GloVe':EmbedInfo(glove_vico_linear_100,False,'glove',100),
        'ViCo':EmbedInfo(glove_vico_linear_100,False,'visual',100),
        'obj_attr': SelectEmbedInfo(glove_vico_select_200,False,'visual',200,'obj_attr'),
        'attr_attr': SelectEmbedInfo(glove_vico_select_200,False,'visual',200,'attr_attr'),
        'context': SelectEmbedInfo(glove_vico_select_200,False,'visual',200,'context'),
        'obj_hyp': SelectEmbedInfo(glove_vico_select_200,False,'visual',200,'obj_hyp'),
    }

    glove_vs_vico_sim.main(exp_const,data_const)

if __name__=='__main__':
    list_exps(globals())