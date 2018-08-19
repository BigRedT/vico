ACTION=$1
GPU=$2

EXP_NAME='trial'
OUT_BASE_DIR="${PWD}/symlinks/exp/semeval_2018_10/concat_svm"

EMBED_LINEAR_FEAT=False
EMBED_QUADRATIC_FEAT=True
DISTANCE_LINEAR_FEAT=True
DISTANCE_QUADRATIC_FEAT=True

echo "Running experiment ${EXP_NAME} ..."

if [[ "${ACTION}" = *"train"* ]]
then
    echo "Initiating training ..."
    CUDA_VISIBLE_DEVICES=$GPU python -m exp.semeval_2018_10.run \
        --exp exp_train_concat_svm \
        --exp_name $EXP_NAME \
        --out_base_dir $OUT_BASE_DIR \
        --embed_linear_feat $EMBED_LINEAR_FEAT \
        --embed_quadratic_feat $EMBED_QUADRATIC_FEAT \
        --distance_linear_feat $DISTANCE_LINEAR_FEAT \
        --distance_quadratic_feat $DISTANCE_QUADRATIC_FEAT
fi

if [[ "${ACTION}" = *"eval"* ]]
then
    echo "Initiating evaluation ..."
    CUDA_VISIBLE_DEVICES=$GPU python -m exp.semeval_2018_10.run \
        --exp exp_eval_concat_svm \
        --exp_name $EXP_NAME \
        --out_base_dir $OUT_BASE_DIR \
        --embed_linear_feat $EMBED_LINEAR_FEAT \
        --embed_quadratic_feat $EMBED_QUADRATIC_FEAT \
        --distance_linear_feat $DISTANCE_LINEAR_FEAT \
        --distance_quadratic_feat $DISTANCE_QUADRATIC_FEAT
fi