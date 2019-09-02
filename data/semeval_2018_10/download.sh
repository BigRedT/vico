#!/bin/bash
# Where the files will be downloaded from
URL="https://github.com/dpaperno/DiscriminAtt.git"

# Where the downloaded files will be stored
TARGET="${PWD}/symlinks/data/semeval_2018_10/raw"

echo "Creating raw data directory ${TARGET}..."
mkdir -p $TARGET

echo "-----------------------------------------------"
echo "Downloading Semeval 2018 Task 10 Repository ..."
echo "-----------------------------------------------"
git clone $URL $TARGET

echo "Preprocessing ..."
bash data/semeval_2018_10/preprocess.sh