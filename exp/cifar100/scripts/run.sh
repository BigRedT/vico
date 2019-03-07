HELD_CLASSES=$1
GPU=$2
bash exp/cifar100/scripts/train.sh $GPU glove 100 $HELD_CLASSES
bash exp/cifar100/scripts/train.sh $GPU glove_random 100 $HELD_CLASSES
bash exp/cifar100/scripts/train.sh $GPU random 100 $HELD_CLASSES
bash exp/cifar100/scripts/train.sh $GPU glove_vico_linear 100 $HELD_CLASSES
bash exp/cifar100/scripts/train.sh $GPU glove_vico_linear 200 $HELD_CLASSES
bash exp/cifar100/scripts/train.sh $GPU glove_vico_select 200 $HELD_CLASSES
bash exp/cifar100/scripts/train.sh $GPU glove_random 200 $HELD_CLASSES