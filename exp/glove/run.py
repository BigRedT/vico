import os

from exp.experimenter import *
from utils.argparse_utils import manage_required_args, str_to_bool
from utils.constants import Constants, ExpConstants
from data.glove.constants import GloveConstantsFactory
import exp.glove.combine_glove_with_visual_features as \
    combine_glove_with_visual_features
import exp.glove.train_ae as train_ae
from exp.glove.models.encoder import EncoderConstants
from exp.glove.models.decoder import DecoderConstants
from exp.glove.concat_embed_dataset import ConcatEmbedDatasetConstants
import exp.glove.save_ae_embeddings as save_ae_embeddings



def exp_combine_glove_with_visual_features():
    exp_name = 'concat_glove_and_visual'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/normalized_resnet_embeddings_recon_loss_trained_on_google')
    exp_const = ExpConstants(exp_name,out_base_dir)

    visual_feat_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/normalized_resnet_features_recon_loss_trained_on_google')
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


def exp_combine_glove_and_visual_features_with_ae():
    exp_name = 'ae_glove_and_visual'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/' + \
        'normalized_resnet_embeddings_recon_loss_trained_on_google')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.batch_size = 10000
    exp_const.lr = 1e-2
    exp_const.num_epochs = 1000

    concat_embeddings_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/' + \
        'normalized_resnet_embeddings_recon_loss_trained_on_google/' + \
        'concat_glove_and_visual')
    data_const = ConcatEmbedDatasetConstants(concat_embeddings_dir)
    data_const.embeddings_h5py = os.path.join(
        data_const.concat_dir,
        'subset_visual_word_vecs.h5py')
    data_const.word_to_idx_json = os.path.join(
        data_const.concat_dir,
        'subset_visual_word_vecs_idx.json')

    model_const = Constants()
    model_const.encoder = EncoderConstants()
    model_const.decoder = DecoderConstants()

    train_ae.main(exp_const,data_const,model_const)


def exp_save_ae_combined_glove_and_visual_features():
    exp_name = 'ae_glove_and_visual'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/' + \
        'normalized_resnet_embeddings_recon_loss_trained_on_google')
    exp_const = ExpConstants(exp_name,out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.batch_size = 1000

    concat_embeddings_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/google_images/' + \
        'normalized_resnet_embeddings_recon_loss_trained_on_google/' + \
        'concat_glove_and_visual')
    data_const = ConcatEmbedDatasetConstants(concat_embeddings_dir)

    model_const = Constants()
    model_const.model_num = 400
    model_const.encoder = EncoderConstants()
    model_const.decoder = DecoderConstants()

    save_ae_embeddings.main(exp_const,data_const,model_const)


if __name__=='__main__':
    list_exps(globals())