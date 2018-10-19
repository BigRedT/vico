ACTION=$1
GPU=$2

EXP_NAME=$3
OUT_BASE_DIR="${PWD}/symlinks/exp/semeval_2018_10/normalized_google_images_recon_loss_visual_word_vecs_trained_on_google"

LR=0.01
L2_WEIGHT=0.001
BATCH_SIZE=2560
GLOVE_DIM=300 #600
EMBED_LINEAR_FEAT=False
EMBED_QUADRATIC_FEAT=False
DISTANCE_LINEAR_FEAT=True
DISTANCE_QUADRATIC_FEAT=True
USE_GLOVE_ONLY=False

if [[ "${USE_GLOVE_ONLY}" = "True" ]]
then
    GLOVE_DIM=300
    EMBEDDINGS_H5PY="${PWD}/symlinks/data/glove/proc/glove_6B_300d.h5py"
    WORD_TO_IDX_JSON="${PWD}/symlinks/data/glove/proc/glove_6B_300d_word_to_idx.json"
else
    EMBEDDINGS_H5PY="${PWD}/symlinks/exp/google_images/normalized_resnet_embeddings_recon_loss_trained_on_google/concat_glove_and_visual/visual_word_vecs.h5py"
    WORD_TO_IDX_JSON="${PWD}/symlinks/exp/google_images/normalized_resnet_embeddings_recon_loss_trained_on_google/concat_glove_and_visual/visual_word_vecs_idx.json"
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
        --distance_quadratic_feat $DISTANCE_QUADRATIC_FEAT \
        --embeddings_h5py $EMBEDDINGS_H5PY \
        --word_to_idx_json $WORD_TO_IDX_JSON \
        --glove_dim $GLOVE_DIM \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --l2_weight $L2_WEIGHT
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
        --word_to_idx_json $WORD_TO_IDX_JSON \
        --glove_dim $GLOVE_DIM \
        --batch_size $BATCH_SIZE
fi