GPU=$1
EMBED_TYPE=$2
GLOVE_DIM=$3
VICO_DIM=$4
HELD_CLASSES=$5
RUN=$6
CUDA_VISIBLE_DEVICES=$GPU python -m exp.cifar100.run \
    --exp exp_train \
    --held_classes $HELD_CLASSES \
    --embed_type $EMBED_TYPE \
    --vico_dim $VICO_DIM \
    --glove_dim $GLOVE_DIM \
    --run $RUN