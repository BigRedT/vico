#!/bin/bash
# Where the files will be downloaded from
BASEURL="https://visualgenome.org/static/data/dataset"
IMG_BASEURL="https://cs.stanford.edu/people/rak248/VG_100K_2"

# Where the downloaded files will be stored
TARGET="${PWD}/symlinks/data/visualgenome/raw"

echo "Creating raw data directory ${TARGET}..."
mkdir -p $TARGET

declare -a ZIP_FILENAMES=(
    "objects.json.zip"
    "attributes.json.zip"
    "relationships.json.zip"
    "object_synsets.json.zip" 
    "attribute_synsets.json.zip" 
    "relationship_synsets.json.zip")

for FILENAME in "${ZIP_FILENAMES[@]}"
do
    echo "-----------------------------------------------"
    echo "Downloading ${FILENAME} ..."
    echo "-----------------------------------------------"
    wget "${BASEURL}/${FILENAME}" -P $TARGET
    unzip "${TARGET}/${FILENAME}" -d $TARGET
    rm "${TARGET}/${FILENAME}"
done

declare -a TXT_FILENAMES=(
    "object_alias.txt"
    "relationship_alias.txt")

for FILENAME in "${TXT_FILENAMES[@]}"
do
    echo "-----------------------------------------------"
    echo "Downloading ${FILENAME} ..."
    echo "-----------------------------------------------"
    wget "${BASEURL}/${FILENAME}" -P $TARGET
done

## Uncomment the following to download images as well (not needed for vico)

# declare -a IMG_DIRNAMES=(
#     "images.zip"
#     "images2.zip")

# for DIRNAME in "${IMG_DIRNAMES[@]}"
# do
#     echo "-----------------------------------------------"
#     echo "Downloading ${DIRNAME} ..."
#     echo "-----------------------------------------------"
#     wget "${IMG_BASEURL}/${DIRNAME}" -P $TARGET
#     unzip "${TARGET}/${DIRNAME}" -d $TARGET
#     rm -rf "${TARGET}/${DIRNAME}"
# done

echo "Preprocessing ..."
bash data/visualgenome/preprocess.sh