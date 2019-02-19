DIM=100
JSONNET="${PWD}/exp/bidaf/bidaf_${DIM}.jsonnet"
OUTDIR="${PWD}/symlinks/exp/bidaf/dim_${DIM}_neg_bias_linear_concat_with_glove_100/run_0"
echo $JSONNET
echo $OUTDIR
allennlp train $JSONNET -s $OUTDIR