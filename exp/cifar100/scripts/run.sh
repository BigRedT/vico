HELD_CLASSES=$1
GPU=$2
RUN=$3
GLOVE_DIM=100
bash exp/cifar100/scripts/train.sh $GPU glove $GLOVE_DIM 0 $HELD_CLASSES $RUN
bash exp/cifar100/scripts/train.sh $GPU vico_linear $GLOVE_DIM 100 $HELD_CLASSES $RUN
bash exp/cifar100/scripts/train.sh $GPU glove_vico_linear $GLOVE_DIM 100 $HELD_CLASSES $RUN
bash exp/cifar100/scripts/train.sh $GPU glove_vico_select $GLOVE_DIM 200 $HELD_CLASSES $RUN