GPU=2
MODE='finetune'
EMBED_DIM=100
XFORM='linear'
FINETUNE_MODEL_NUM=80000

echo "------------------------------------------------------------"
echo "ViCo specifications:"
echo "-Transformation: ${XFORM}"
echo "-Embedding dimension: ${EMBED_DIM}"
echo "------------------------------------------------------------"

if [[ "${MODE}" = *"train"* ]]
then
    echo "------------------------------------------------------------"
    echo "Training ViCo on GPU #${GPU} ..."
    echo "------------------------------------------------------------"
    CUDA_VISIBLE_DEVICES=$GPU python \
        -m exp.multi_sense_cooccur.run \
        --exp exp_train \
        --embed_dim $EMBED_DIM \
        --xform $XFORM
fi

if [[ "${MODE}" = *"finetune"* ]]
then
    echo "------------------------------------------------------------"
    echo "Finetuning ViCo on GPU #${GPU} from model number ${FINETUNE_MODEL_NUM} ..."
    echo "------------------------------------------------------------"
    CUDA_VISIBLE_DEVICES=$GPU python \
        -m exp.multi_sense_cooccur.run \
        --exp exp_train \
        --embed_dim $EMBED_DIM \
        --xform $XFORM \
        --model_num $FINETUNE_MODEL_NUM
fi
