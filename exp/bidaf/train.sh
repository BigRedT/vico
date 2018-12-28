JSONNET="${PWD}/exp/bidaf/bidaf.jsonnet"
OUTDIR="${PWD}/symlinks/exp/bidaf/cooccur_gt/concat"
echo $JSONNET
echo $OUTDIR
allennlp train $JSONNET -s $OUTDIR