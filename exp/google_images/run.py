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
import exp.google_images.train as train
from data.semeval_2018_10.constants import SemEval201810Constants
import utils.io as io


def exp_cache_resnet_features():
    exp_name = 'normalized_resnet_features_trained_on_google'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images')
    exp_const = ExpConstants(exp_name=exp_name,out_base_dir=out_base_dir)
    exp_const.use_resnet_normalized=True
    exp_const.model_path = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/train_resnet_normalized_google_images/models/net_33000')
    exp_const.feature_dim = 2048
    
    # Compute vocabulary for semeval2018
    semeval_const = SemEval201810Constants()
    data_const = GoogleImagesDatasetConstants()
    data_const.vocab_json = semeval_const.all_word_freqs

    cache_resnet_features.main(exp_const,data_const)


def exp_train_resnet():
    exp_name = 'train_resnet_normalized_google_images'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.batch_size = 32
    exp_const.num_epochs = 25
    exp_const.lr = 0.1 #1e-3
    exp_const.momentum = 0.9
    exp_const.num_workers = 5
    exp_const.load_finetuned = True
    exp_const.train_net = True # set True to train the backbone (resnet)
    exp_const.optimizer = 'SGD'
    exp_const.use_resnet_normalized = True

    semeval_const = SemEval201810Constants()
    data_const = GoogleImagesImageLevelDatasetConstants()
    data_const.vocab_json = semeval_const.all_word_freqs
    num_words = len(io.load_json_object(data_const.vocab_json))

    model_const = Constants()
    model_const.model_num = 66000
    model_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/train_resnet_normalized_google_images/models')
    model_const.net_path = os.path.join(
        model_dir,
        f'net_{model_const.model_num}')
    model_const.word_classifier_layers_path = os.path.join(
        model_dir,
        f'word_classifier_layers_{model_const.model_num}')
    model_const.word_classifier_layers = WordClassifierLayerConstants()
    model_const.word_classifier_layers.num_classes = num_words
    model_const.word_classifier_layers.layer_units = []

    train.main(exp_const,data_const,model_const)


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