# Glove
# VISUAL_WORD_VECS_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/data/glove/proc"
# VISUAL_WORD_VECS_H5PY="${VISUAL_WORD_VECS_DIR}/glove_6B_300d.h5py"
# VISUAL_WORD_VECS_IDX_JSON="${VISUAL_WORD_VECS_DIR}/glove_6B_300d_word_to_idx.json"

VISUAL_WORD_VECS_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/\
exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms/\
dim_200_neg_bias_linear/concat_with_glove_300"
VISUAL_WORD_VECS_H5PY="${VISUAL_WORD_VECS_DIR}/visual_word_vecs.h5py"
VISUAL_WORD_VECS_IDX_JSON="${VISUAL_WORD_VECS_DIR}/visual_word_vecs_idx.json"
VISUAL_WORDS_JSON="${VISUAL_WORD_VECS_DIR}/visual_words.json"
GLOVE_DIM=300

VQA_VOCAB_TXT="/home/nfs/tgupta6/Code/pythia/data/vocabulary_vqa.txt"

OUTDIR="/home/nfs/tgupta6/Code/pythia/data"
VQA_WORD_VECS_NPY="${OUTDIR}/vqa2.200_neg_bias_linear_concat_with_glove_300.npy"

# Comment out visual_words_json when using glove
python -m exp.pythia.write_vqa_embeddings \
    --visual_word_vecs_h5py $VISUAL_WORD_VECS_H5PY \
    --visual_word_vecs_idx_json $VISUAL_WORD_VECS_IDX_JSON \
    --visual_words_json $VISUAL_WORDS_JSON \
    --vqa_vocab_txt $VQA_VOCAB_TXT \
    --vqa_word_vecs_npy $VQA_WORD_VECS_NPY \
    --glove_dim $GLOVE_DIM