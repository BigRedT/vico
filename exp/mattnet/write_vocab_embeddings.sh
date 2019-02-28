MATTNET_DATA_JSON="/home/nfs/tgupta6/Code/MAttNet/cache/prepro/refcoco+_unc/data.json"

# Glove
# VISUAL_WORD_VECS_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/data/glove/proc"
# VISUAL_WORD_VECS_H5PY="${VISUAL_WORD_VECS_DIR}/glove_6B_300d.h5py"
# VISUAL_WORD_VECS_IDX_JSON="${VISUAL_WORD_VECS_DIR}/glove_6B_300d_word_to_idx.json"

VISUAL_WORD_VECS_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms/dim_200_neg_bias_select/concat_with_glove_300"
VISUAL_WORD_VECS_H5PY="${VISUAL_WORD_VECS_DIR}/visual_word_vecs.h5py"
VISUAL_WORD_VECS_IDX_JSON="${VISUAL_WORD_VECS_DIR}/visual_word_vecs_idx.json"
VISUAL_WORDS_JSON="${VISUAL_WORD_VECS_DIR}/visual_words.json"


MATTNET_WORD_VECS_NPY="/home/nfs/tgupta6/Code/MAttNet/cache/prepro/refcoco+_unc/dim_200_neg_bias_select_concat_with_glove_300.npy"

# Comment out visual_words_json when using glove
python -m exp.mattnet.write_vocab_embeddings \
    --mattnet_data_json $MATTNET_DATA_JSON \
    --mattnet_word_vecs_npy $MATTNET_WORD_VECS_NPY \
    --visual_word_vecs_h5py $VISUAL_WORD_VECS_H5PY \
    --visual_word_vecs_idx_json $VISUAL_WORD_VECS_IDX_JSON \
    --visual_words_json $VISUAL_WORDS_JSON 
