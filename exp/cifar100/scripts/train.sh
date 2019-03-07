GPU=$1
HELD_CLASSES=$4
EMBED_TYPE=$2
VICO_DIM=$3
CUDA_VISIBLE_DEVICES=$GPU python -m exp.cifar100.run \
    --exp exp_train \
    --held_classes $HELD_CLASSES \
    --embed_type $EMBED_TYPE \
    --vico_dim $VICO_DIM