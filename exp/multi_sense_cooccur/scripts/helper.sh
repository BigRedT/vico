GPU=$CUDA_VISIBLE_DEVICES
# Include any combination of 'cooccur', 'train', 'finetune', 'extract' in the MODE string 
# to perform the corresponding steps. For example 'train_extract' would train 
# ViCo model, skip finetuning, and extract embeddings. 
MODE='train_finetune_extract_concat_tsne' #'cooccur_train_finetune_extract_concat_tsne'
EMBED_DIM=100
XFORM='linear'
# FINETUNE_MODEL_NUM and MODEL_NUM must correspond to one of the saved models
# which will need to be loaded to continue finetuning or extract embeddings
FINETUNE_MODEL_NUM=80000 
MODEL_NUM=160000
GLOVE_DIM=300 # For concatenating with ViCo
SYN=False
# Set SYN to true to use Synonym co-occurrences during training. The numbers in the 
# paper are without using synonym co-occurrences because we empirically found it
# to hurt performance in some cases.

export HDF5_USE_FILE_LOCKING=FALSE

if [[ "${MODE}" = *"cooccur"* ]]
then
    echo "------------------------------------------------------------"
    echo "Computing cooccurrences ..."
    echo "------------------------------------------------------------"
    bash exp/multi_sense_cooccur/create_cooccur.sh
fi

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
        --xform $XFORM \
        --syn $SYN
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
        --model_num $FINETUNE_MODEL_NUM \
        --syn $SYN
fi

if [[ "${MODE}" = *"extract"* ]]
then
    echo "------------------------------------------------------------"
    echo "Extracting ViCo embeddings from model number ${MODEL_NUM} ..."
    echo "------------------------------------------------------------"
    python \
        -m exp.multi_sense_cooccur.run \
        --exp exp_extract_embeddings \
        --embed_dim $EMBED_DIM \
        --xform $XFORM \
        --model_num $MODEL_NUM \
        --syn $SYN
fi

if [[ "${MODE}" = *"concat"* ]]
then
    echo "------------------------------------------------------------"
    echo "Concatenating ViCo embeddings with ${GLOVE_DIM} dim. GloVe ..."
    echo "------------------------------------------------------------"
    python \
        -m exp.multi_sense_cooccur.run \
        --exp exp_concat_with_glove \
        --embed_dim $EMBED_DIM \
        --xform $XFORM \
        --glove_dim $GLOVE_DIM
fi

if [[ "${MODE}" = *"tsne"* ]]
then
    echo "------------------------------------------------------------"
    echo "Visualizing coarse category clustering using t-SNE ..."
    echo "------------------------------------------------------------"
    python \
        -m exp.multi_sense_cooccur.run \
        --exp exp_vis_pca_tsne \
        --embed_dim $EMBED_DIM \
        --xform $XFORM \
        --glove_dim $GLOVE_DIM
fi