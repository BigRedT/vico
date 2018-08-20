ACTION=$1
GPU=$2

EXP_NAME='glove_rerun3'
OUT_BASE_DIR="${PWD}/symlinks/exp/semeval_2018_10/concat_svm_random_embeddings"

EMBED_LINEAR_FEAT=False
EMBED_QUADRATIC_FEAT=True
DISTANCE_LINEAR_FEAT=True
DISTANCE_QUADRATIC_FEAT=True
EMBEDDINGS_H5PY="${PWD}/symlinks/data/glove/proc/glove_6B_300d.h5py"
WORD_TO_IDX_JSON="${PWD}/symlinks/data/glove/proc/glove_6B_300d_word_to_idx.json"

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
        --distance_quadratic_feat $DISTANCE_QUADRATIC_FEAT \
        --embeddings_h5py $EMBEDDINGS_H5PY \
        --word_to_idx_json $WORD_TO_IDX_JSON
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
        --distance_quadratic_feat $DISTANCE_QUADRATIC_FEAT \
        --embeddings_h5py $EMBEDDINGS_H5PY \
        --word_to_idx_json $WORD_TO_IDX_JSON
fi