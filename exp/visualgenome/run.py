import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from data.glove.constants import GloveConstantsFactory
from data.visualgenome.constants import VisualGenomeConstants
from exp.visualgenome.image_regions_dataset import ImageRegionsDatasetConstants
from exp.visualgenome.models.word_classification_mlp import \
    WordClassifierLayerConstants
import exp.visualgenome.vis.visualize_regions as visualize_regions
import exp.visualgenome.cache_resnet_features as cache_resnet_features
import exp.visualgenome.collect_visual_features as collect_visual_features
import exp.visualgenome.combine_glove_with_visual_features as \
    combine_glove_with_visual_features
import exp.visualgenome.train as train
import exp.visualgenome.eval as evaluation

def exp_visualize_regions():
    exp_name = 'region_visualization'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.num_batches = 10
    exp_const.batch_size = 5
    exp_const.shuffle = False
    exp_const.num_workers = 5
    
    data_const = ImageRegionsDatasetConstants()

    visualize_regions.main(exp_const,data_const)


def exp_cache_resnet_features():
    exp_name = 'resnet_features_normalized_finetuned'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.batch_size = 5
    exp_const.num_workers = 5
    exp_const.feature_dim = 2048 # 2208 for densenet161, 1024 for densent 121, 2048 for resnet152
    exp_const.use_resnet_normalized = True
    exp_const.model_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome/train_resnet_normalized_sgd_finetune_260500/models/net_1244500')
    data_const = ImageRegionsDatasetConstants()
    data_const.imagenet_labels_json = os.path.join(
        os.getcwd(),
        'symlinks/data/imagenet/proc/labels.json')

    cache_resnet_features.main(exp_const,data_const)


def exp_collect_visual_features():
    exp_name = 'finetuned_resnet_normalized_embeddings'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome')
    exp_const = ExpConstants(exp_name,out_base_dir)
    
    data_const = VisualGenomeConstants()
    data_const.visual_features_h5py = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome/resnet_features_normalized_finetuned/' + \
        'visual_features.h5py')
    data_const.object_id_to_features_idx_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome/resnet_features_normalized_finetuned/' + \
        'object_id_to_features_idx.json')
    
    collect_visual_features.main(exp_const,data_const)


def exp_combine_glove_with_visual_features():
    exp_name = 'concat_glove_and_visual'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome/finetuned_resnet_normalized_embeddings')
    exp_const = ExpConstants(exp_name,out_base_dir)

    visual_feat_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome/finetuned_resnet_normalized_embeddings')
    data_const = Constants()
    data_const.visual_features_idx = os.path.join(
        visual_feat_dir,
        'word_to_idx.json')
    data_const.visual_features_h5py = os.path.join(
        visual_feat_dir,
        'word_features.h5py')
    glove_const = GloveConstantsFactory.create()
    data_const.glove_idx = glove_const.word_to_idx_json
    data_const.glove_h5py = glove_const.embeddings_h5py

    combine_glove_with_visual_features.main(exp_const,data_const)


def exp_train_resnet():
    exp_name = 'train_resnet_normalized_finetune_249k'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.batch_size = 5
    exp_const.region_batch_size = 32
    exp_const.num_epochs = 15
    exp_const.lr = 0.01 #1e-3
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.load_finetuned = True
    exp_const.train_net = True # set True to train the backbone (resnet)
    exp_const.optimizer = 'SGD'
    exp_const.use_resnet_normalized = True

    data_const = ImageRegionsDatasetConstants()

    model_const = Constants()
    model_const.model_num = 249000
    model_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome/train_resnet_normalized/models')
    model_const.net_path = os.path.join(
        model_dir,
        f'net_{model_const.model_num}')
    model_const.object_classifier_layers_path = os.path.join(
        model_dir,
        f'object_classifier_layers_{model_const.model_num}')
    model_const.attribute_classifier_layers_path = os.path.join(
        model_dir,
        f'attribute_classifier_layers_{model_const.model_num}')
    model_const.object_classifier_layers = WordClassifierLayerConstants()
    model_const.object_classifier_layers.num_classes = 8078
    model_const.object_classifier_layers.layer_units = []
    model_const.attribute_classifier_layers = WordClassifierLayerConstants()
    model_const.attribute_classifier_layers.num_classes = 6497
    model_const.attribute_classifier_layers.layer_units = []

    train.main(exp_const,data_const,model_const)


def exp_eval_resnet():
    exp_name = 'train_resnet_normalized'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/visualgenome')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.batch_size = 5
    exp_const.region_batch_size = 32
    exp_const.num_workers = 5
    exp_const.num_eval_iters = 1000
    exp_const.use_resnet_normalized = True

    data_const = ImageRegionsDatasetConstants()

    model_const = Constants()
    model_const.model_num = 249000
    model_dir = exp_const.model_dir
    model_const.net_path = os.path.join(
        model_dir,
        f'net_{model_const.model_num}')
    model_const.object_classifier_layers_path = os.path.join(
        model_dir,
        f'object_classifier_layers_{model_const.model_num}')
    model_const.attribute_classifier_layers_path = os.path.join(
        model_dir,
        f'attribute_classifier_layers_{model_const.model_num}')
    model_const.object_classifier_layers = WordClassifierLayerConstants()
    model_const.object_classifier_layers.num_classes = 8078
    model_const.object_classifier_layers.layer_units = []
    model_const.attribute_classifier_layers = WordClassifierLayerConstants()
    model_const.attribute_classifier_layers.num_classes = 6497
    model_const.attribute_classifier_layers.layer_units = []

    evaluation.main(exp_const,data_const,model_const)

if __name__=='__main__':
    list_exps(globals())
