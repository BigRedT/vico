VOCAB_PKL="/home/nfs/tgupta6/Code/vsepp/vocab/coco_precomp_vocab.pkl"

# Glove
# VISUAL_WORD_VECS_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/data/glove/proc"
# VISUAL_WORD_VECS_H5PY="${VISUAL_WORD_VECS_DIR}/glove_6B_300d.h5py"
# VISUAL_WORD_VECS_IDX_JSON="${VISUAL_WORD_VECS_DIR}/glove_6B_300d_word_to_idx.json"

VISUAL_WORD_VECS_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/\
exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms/\
dim_100_neg_bias_linear/concat_with_glove_300"
#VISUAL_WORD_VECS_H5PY="${VISUAL_WORD_VECS_DIR}/visual_word_vecs.h5py"
VISUAL_WORD_VECS_H5PY="${VISUAL_WORD_VECS_DIR}/glove_random_word_vecs.h5py"
VISUAL_WORD_VECS_IDX_JSON="${VISUAL_WORD_VECS_DIR}/visual_word_vecs_idx.json"
VISUAL_WORDS_JSON="${VISUAL_WORD_VECS_DIR}/visual_words.json"


# VSEPP_WORD_VECS_NPY="/home/nfs/tgupta6/Code/vsepp_data/data/coco_precomp/\
# dim_200_neg_bias_linear_concat_with_glove_300.npy"
VSEPP_WORD_VECS_NPY="/home/nfs/tgupta6/Code/vsepp_data/data/coco_precomp/\
dim_100_random_concat_with_glove_300.npy"

# Comment out visual_words_json when using glove
python -m exp.vsepp.write_vocab_embeddings \
    --vocab_pkl $VOCAB_PKL \
    --vsepp_word_vecs_npy $VSEPP_WORD_VECS_NPY \
    --visual_word_vecs_h5py $VISUAL_WORD_VECS_H5PY \
    --visual_word_vecs_idx_json $VISUAL_WORD_VECS_IDX_JSON \
    --visual_words_json $VISUAL_WORDS_JSON 
