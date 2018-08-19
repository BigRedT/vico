SELECT_EXP=$1
ACTION=$2
GPU=$3

OUT_BASE_DIR="${PWD}/symlinks/exp/semeval_2018_10/concat_svm_feature_ablation_glove"

EMBED_LINEAR_FEAT=True
EMBED_QUADRATIC_FEAT=True
DISTANCE_LINEAR_FEAT=True
DISTANCE_QUADRATIC_FEAT=True

if [ "${SELECT_EXP}" = "all_features" ]
then
    EXP_NAME="embed_linear_quad_distance_linear_quad"
    

elif [ "${SELECT_EXP}" = "no_embed_linear" ]
then
    EXP_NAME="embed_quad_distance_linear_quad"
    EMBED_LINEAR_FEAT=False

elif [ "${SELECT_EXP}" = "no_embed_quad" ]
then
    EXP_NAME="embed_linear_distance_linear_quad"
    EMBED_QUADRATIC_FEAT=False

elif [ "${SELECT_EXP}" = "no_embed" ]
then
    EXP_NAME="distance_linear_quad"
    EMBED_LINEAR_FEAT=False
    EMBED_QUADRATIC_FEAT=False

elif [ "${SELECT_EXP}" = "no_distance_linear" ]
then
    EXP_NAME="embed_linear_quad_distance_quad"
    DISTANCE_LINEAR_FEAT=False

elif [ "${SELECT_EXP}" = "no_distance_quad" ]
then
    EXP_NAME="embed_linear_quad_distance_linear"
    DISTANCE_QUADRATIC_FEAT=False

elif [ "${SELECT_EXP}" = "no_distance" ]
then
    EXP_NAME="embed_linear_quad"
    DISTANCE_LINEAR_FEAT=False
    DISTANCE_QUADRATIC_FEAT=False

else
    echo "Please select an experiment to run!"
    exit 1
fi

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
