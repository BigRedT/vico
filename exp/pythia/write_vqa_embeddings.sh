VISUAL_WORD_VECS_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/exp/cooccur/imagenet_genome_gt/training_no_fx_self_count_dim_50_single_embed/concat_with_glove_300"
VISUAL_WORD_VECS_H5PY="${VISUAL_WORD_VECS_DIR}/visual_word_vecs.h5py"
VISUAL_WORD_VECS_IDX_JSON="${VISUAL_WORD_VECS_DIR}/visual_word_vecs_idx.json"
VISUAL_WORDS_JSON="${VISUAL_WORD_VECS_DIR}/visual_words.json"
GLOVE_DIM=300

VQA_VOCAB_TXT="/data/tanmay/pythia/data/vocabulary_vqa.txt"

OUTDIR="/data/tanmay/pythia/data"
VQA_WORD_VECS_NPY="${OUTDIR}/vqa2.training_no_fx_self_count_dim_50_single_embed_concat_with_glove_300.npy"

python -m exp.pythia.write_vqa_embeddings \
    --visual_word_vecs_h5py $VISUAL_WORD_VECS_H5PY \
    --visual_word_vecs_idx_json $VISUAL_WORD_VECS_IDX_JSON \
    --visual_words_json $VISUAL_WORDS_JSON \
    --vqa_vocab_txt $VQA_VOCAB_TXT \
    --vqa_word_vecs_npy $VQA_WORD_VECS_NPY \
    --glove_dim $GLOVE_DIM