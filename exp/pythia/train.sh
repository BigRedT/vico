OUT_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/exp/pythia"
echo $OUT_DIR

# # Model with visual embedding
# CUDA_VISIBLE_DEVICES=2 python train.py --out_dir $OUT_DIR --seed 0 --config_overwrite '{data:{image_fast_reader:false}}' --config 'config/verbose/training_no_fx_self_count_dim_50_single_embed_concat_with_glove_300.yaml'

# # Default model
CUDA_VISIBLE_DEVICES=1 python train.py --out_dir $OUT_DIR --seed 0 --config_overwrite '{data:{image_fast_reader:false}}'
