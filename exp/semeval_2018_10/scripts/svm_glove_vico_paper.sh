ACTION=$1
RUN=$2

GPU=$CUDA_VISIBLE_DEVICES

export HDF5_USE_FILE_LOCKING=FALSE

EXP_NAME="run_${RUN}"
OUT_BASE_DIR="${PWD}/symlinks/exp/semeval_2018_10/glove_300_vico_linear_100_paper"

LR=0.1 # 0.01
L2_WEIGHT=0.001 # lamba/2
BATCH_SIZE=2560
GLOVE_DIM=300
EMBED_LINEAR_FEAT=False
EMBED_QUADRATIC_FEAT=False
DISTANCE_LINEAR_FEAT=True
DISTANCE_QUADRATIC_FEAT=True
USE_GLOVE_ONLY=False
USE_VISUAL_ONLY=False

# Note -  For random set USE_GLOVE_ONLY to True and self.random=True 
# in SemEval201810DatasetConstants class in dataset.py file

if [[ "${USE_GLOVE_ONLY}" = "True" ]]
then
    GLOVE_DIM=300
    EMBEDDINGS_H5PY="${PWD}/symlinks/data/glove/proc/glove_6B_300d.h5py"
    WORD_TO_IDX_JSON="${PWD}/symlinks/data/glove/proc/glove_6B_300d_word_to_idx.json"
    VISUAL_VOCAB_JSON="${PWD}/symlinks/exp/multi_sense_cooccur/linear_100/concat_with_glove_300/visual_words.json"
else
    # Change these paths to point to where the pretrained embeddings are downloaded
    EMBEDDINGS_H5PY="${PWD}/symlinks/exp/multi_sense_cooccur/paper/linear_100/visual_word_vecs.h5py"
    WORD_TO_IDX_JSON="${PWD}/symlinks/exp/multi_sense_cooccur/paper/linear_100/visual_word_vecs_idx.json"
    VISUAL_VOCAB_JSON="${PWD}/symlinks/exp/multi_sense_cooccur/paper/linear_100/visual_words.json"
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
        --visual_only $USE_VISUAL_ONLY \
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
        --visual_only $USE_VISUAL_ONLY \
        --embeddings_h5py $EMBEDDINGS_H5PY \
        --word_to_idx_json $WORD_TO_IDX_JSON \
        --visual_vocab_json $VISUAL_VOCAB_JSON \
        --glove_dim $GLOVE_DIM \
        --batch_size $BATCH_SIZE
fi