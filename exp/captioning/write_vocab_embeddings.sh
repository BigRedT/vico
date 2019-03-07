COCOTALK_JSON="/home/nfs/tgupta6/Code/ImageCaptioning.pytorch/data/cocotalk.json"
#COCOTALK_WORD_VECS_NPY="/home/nfs/tgupta6/Code/ImageCaptioning.pytorch/data/embeddings/dim_200_neg_bias_linear_concat_with_glove_300"
COCOTALK_WORD_VECS_NPY="/home/nfs/tgupta6/Code/ImageCaptioning.pytorch/data/embeddings/dim_100_random_concat_with_glove_300"
USE_GLOVE=False

if [[ "${USE_GLOVE}" = "True" ]]
then
    # Glove
    VISUAL_WORD_VECS_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/data/glove/proc"
    VISUAL_WORD_VECS_H5PY="${VISUAL_WORD_VECS_DIR}/glove_6B_300d.h5py"
    VISUAL_WORD_VECS_IDX_JSON="${VISUAL_WORD_VECS_DIR}/glove_6B_300d_word_to_idx.json"
    COCOTALK_WORD_VECS_NPY="/home/nfs/tgupta6/Code/ImageCaptioning.pytorch/data/embeddings/glove_300"

    python -m exp.captioning.write_vocab_embeddings \
        --cocotalk_json $COCOTALK_JSON \
        --cocotalk_word_vecs_npy $COCOTALK_WORD_VECS_NPY \
        --visual_word_vecs_h5py $VISUAL_WORD_VECS_H5PY \
        --visual_word_vecs_idx_json $VISUAL_WORD_VECS_IDX_JSON

else
    VISUAL_WORD_VECS_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms/dim_100_neg_bias_linear/concat_with_glove_300"
    #VISUAL_WORD_VECS_H5PY="${VISUAL_WORD_VECS_DIR}/visual_word_vecs.h5py"
    VISUAL_WORD_VECS_H5PY="${VISUAL_WORD_VECS_DIR}/glove_random_word_vecs.h5py"
    VISUAL_WORD_VECS_IDX_JSON="${VISUAL_WORD_VECS_DIR}/visual_word_vecs_idx.json"
    VISUAL_WORDS_JSON="${VISUAL_WORD_VECS_DIR}/visual_words.json"

    python -m exp.captioning.write_vocab_embeddings \
        --cocotalk_json $COCOTALK_JSON \
        --cocotalk_word_vecs_npy $COCOTALK_WORD_VECS_NPY \
        --visual_word_vecs_h5py $VISUAL_WORD_VECS_H5PY \
        --visual_word_vecs_idx_json $VISUAL_WORD_VECS_IDX_JSON \
        --visual_words_json $VISUAL_WORDS_JSON 
fi