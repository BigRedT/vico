import os
import copy
from tqdm import tqdm

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
import exp.google_images.cache_resnet_features as cache_resnet_features
from exp.google_images.dataset import GoogleImagesDatasetConstants
from exp.google_images.dataset_image_level import \
    GoogleImagesImageLevelDatasetConstants
from exp.google_images.models.word_classification_mlp import \
    WordClassifierLayerConstants
from exp.google_images.models.decoder import DecoderConstants
import exp.google_images.train as train
import exp.google_images.eval as evaluator
import exp.google_images.vis_feature_content as vis_feature_content
from data.semeval_2018_10.constants import SemEval201810Constants
import utils.io as io


def exp_cache_resnet_features():
    exp_name = 'normalized_resnet_features_recon_loss_trained_on_google'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images')
    exp_const = ExpConstants(exp_name=exp_name,out_base_dir=out_base_dir)
    exp_const.use_resnet_normalized=True
    exp_const.model_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/train_resnet_normalized_recon_loss_google_images/models/net_122000')
    exp_const.feature_dim = 2048
    
    # Compute vocabulary for semeval2018
    semeval_const = SemEval201810Constants()
    data_const = GoogleImagesDatasetConstants()
    data_const.vocab_json = semeval_const.all_word_freqs

    cache_resnet_features.main(exp_const,data_const)


def exp_train_resnet():
    exp_name = 'train_resnet_normalized_recon_loss_hf_google_images'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.batch_size = 32
    exp_const.num_epochs = 10000
    exp_const.lr = 0.1 #1e-3
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.load_finetuned = False
    exp_const.train_net = True # set True to train the backbone (resnet)
    exp_const.optimizer = 'SGD'
    exp_const.use_resnet_normalized = True
    exp_const.use_recon_loss = True

    semeval_const = SemEval201810Constants()
    data_const = GoogleImagesImageLevelDatasetConstants()
    data_const.vocab_json = semeval_const.all_word_freqs
    num_words = len(io.load_json_object(data_const.vocab_json))

    model_const = Constants()
    model_const.model_num = None
    model_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/train_resnet_normalized_recon_loss_hf_google_images/models')
    model_const.net_path = os.path.join(
        model_dir,
        f'net_{model_const.model_num}')
    model_const.word_classifier_layers_path = os.path.join(
        model_dir,
        f'word_classifier_layers_{model_const.model_num}')
    model_const.word_classifier_layers = WordClassifierLayerConstants()
    model_const.word_classifier_layers.num_classes = num_words
    model_const.word_classifier_layers.layer_units = []
    model_const.decoder = DecoderConstants()

    train.main(exp_const,data_const,model_const)


def exp_eval_resnet():
    exp_name = 'train_resnet_normalized_recon_loss_google_images'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.batch_size = 64
    exp_const.num_workers = 5
    exp_const.use_resnet_normalized = True

    semeval_const = SemEval201810Constants()
    data_const = GoogleImagesImageLevelDatasetConstants()
    data_const.vocab_json = semeval_const.all_word_freqs
    num_words = len(io.load_json_object(data_const.vocab_json))

    model_const = Constants()
    model_const.model_num = 122000
    model_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/train_resnet_normalized_recon_loss_google_images/models')
    model_const.net_path = os.path.join(
        model_dir,
        f'net_{model_const.model_num}')
    model_const.word_classifier_layers_path = os.path.join(
        model_dir,
        f'word_classifier_layers_{model_const.model_num}')
    model_const.word_classifier_layers = WordClassifierLayerConstants()
    model_const.word_classifier_layers.num_classes = num_words
    model_const.word_classifier_layers.layer_units = []

    evaluator.main(exp_const,data_const,model_const)


def exp_vis_feature_content_resnet():
    exp_name = 'train_resnet_normalized_recon_loss_google_images'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.deepdream_dir = os.path.join(exp_const.exp_dir,'deepdream')
    exp_const.batch_size = 8
    exp_const.num_workers = 5
    exp_const.use_resnet_normalized = True
    exp_const.num_gen_imgs = 8
    exp_const.max_inner_iter = 1
    exp_const.lr = 1e6 #1e-3
    exp_const.momentum = 0.9

    semeval_const = SemEval201810Constants()
    data_const = GoogleImagesImageLevelDatasetConstants()
    data_const.vocab_json = semeval_const.all_word_freqs
    num_words = len(io.load_json_object(data_const.vocab_json))

    model_const = Constants()
    model_const.model_num = 122000
    model_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/train_resnet_normalized_recon_loss_google_images/models')
    model_const.net_path = os.path.join(
        model_dir,
        f'net_{model_const.model_num}')
    model_const.word_classifier_layers_path = os.path.join(
        model_dir,
        f'word_classifier_layers_{model_const.model_num}')
    model_const.word_classifier_layers = WordClassifierLayerConstants()
    model_const.word_classifier_layers.num_classes = num_words
    model_const.word_classifier_layers.layer_units = []

    vis_feature_content.main(exp_const,data_const,model_const)


def exp_delete_models():
    models_dir = '/data/tanmay/visual_word_vecs/exp/google_images/train_resnet_normalized_google_images/models'
    until_model = 60000
    for model_num in tqdm(range(0,until_model)):
        for name in ['net','word_classifier_layers']:
            model_file = os.path.join(models_dir,f'{name}_{model_num}')
            if os.path.exists(model_file):
                os.remove(model_file)



if __name__=='__main__':
    list_exps(globals())