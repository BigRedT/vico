SRC_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/effect_of_xforms"
S3_BUCKET="s3://vico-tanmay"

# Upload cooccurrences
cooccur_csv="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/merged_cooccur_self.csv"
aws s3 cp $cooccur_csv $S3_BUCKET

src_path_s3="${S3_BUCKET}/merged_cooccur_self.csv"
tgt_path_s3="${S3_BUCKET}/public_data/cooccur.csv"
aws s3 mv $src_path_s3 $tgt_path_s3

# Upload embeddings
models_src_tgt=( 
    "dim_50_neg_bias_linear,glove_300_vico_linear_50"
    #"dim_100_neg_bias_linear,glove_300_vico_linear_100"
    "dim_200_neg_bias_linear,glove_300_vico_linear_200"
    #"dim_200_neg_bias_select,glove_300_vico_select_200" 
)
files=(
    "visual_word_vecs.h5py"
    "visual_word_vecs_idx.json"
    "visual_words.json"
)
for model_src_tgt in "${models_src_tgt[@]}"
do
    tmp=(${model_src_tgt//,/ })
    model_src=${tmp[0]}
    model_tgt=${tmp[1]}
    for file in "${files[@]}"
    do
        src_path="${SRC_DIR}/${model_src}/concat_with_glove_300/${file}"
        aws s3 cp $src_path $S3_BUCKET

        src_path_s3="${S3_BUCKET}/${file}"
        tgt_path_s3="${S3_BUCKET}/public_data/${model_tgt}/${file}"
        aws s3 mv $src_path_s3 $tgt_path_s3
    done
done
