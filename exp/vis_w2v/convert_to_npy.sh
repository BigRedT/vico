TYPE="visual_word2vec_wiki"
#TYPE="word2vec_wiki"
EMBED_PATH="/home/nfs/tgupta6/Code/visual_word2vec/${TYPE}.txt"
OUT_DIR="${PWD}/symlinks/exp/vis_w2v/${TYPE}"
VOCAB_JSON="${PWD}/symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms/dim_100_neg_bias_linear/word_to_idx.json"
echo "Embed Path: ${EMBED_PATH}"
echo "Output Dir: ${OUT_DIR}"
python -m exp.vis_w2v.convert_to_npy $EMBED_PATH $OUT_DIR $VOCAB_JSON $TYPE 