TRAIN_URL="https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-train-v1.1.json"
DEV_URL="https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json"
OUT_DIR="${PWD}/symlinks/data/squad_v1_1"

mkdir $OUT_DIR

echo "Downloading training data from ${TRAIN_URL} ..."
wget --directory-prefix=$OUT_DIR $TRAIN_URL

echo "Downloading dev data from ${DEV_URL} ..."
wget --directory-prefix=$OUT_DIR $DEV_URL