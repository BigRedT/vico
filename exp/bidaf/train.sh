JSONNET="${PWD}/exp/bidaf/bidaf.jsonnet"
OUTDIR="${PWD}/symlinks/exp/bidaf/concat_glove_visual_avg_reps_balanced_bce_norm1/glove_visual"
echo $JSONNET
echo $OUTDIR
allennlp train $JSONNET -s $OUTDIR