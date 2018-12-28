EXP_DIR="/home/nfs/tgupta6/Code/visual_word_vecs/symlinks/exp/pythia/results/training_no_fx_self_count_dim_50_single_embed_concat_with_glove_300/data.image_fast_reader.false_34920"
CONFIG="${EXP_DIR}/config.yaml"
MODEL_PATH="${best_model.pth}"
OUT_PREFIX="test_best_model"

python run_test.py \
    --config $CONFIG \
    --model_path $MODEL_PATH \
    --out_prefix $OUT_PREFIX
