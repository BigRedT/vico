# echo "Downloading glove.6B.zip file ..."
# URL="http://nlp.stanford.edu/data/glove.6B.zip"
# OUT_DIR="${PWD}/symlinks/data/glove"
wget --directory-prefix=$OUT_DIR $URL

echo "Unzipping glove.6B.zip to glove.6B ..."
GLOVE_ZIP="${OUT_DIR}/glove.6B.zip"
GLOVE_DIR="${OUT_DIR}/glove.6B"
mkdir $GLOVE_DIR
unzip $GLOVE_ZIP -d $GLOVE_DIR
rm $GLOVE_ZIP


export HDF5_USE_FILE_LOCKING=FALSE

echo "Save glove 100 dim. as hdf5 ..."
python -m data.glove.save_as_hdf5 \
    --glove_txt "${PWD}/symlinks/data/glove/glove.6B/glove.6B.100d.txt"
    --name "glove_6B_100d"

echo "Save glove 300 dim. as hdf5 ..."
python -m data.glove.save_as_hdf5 \
    --glove_txt "${PWD}/symlinks/data/glove/glove.6B/glove.6B.300d.txt" \
    --name "glove_6B_300d"