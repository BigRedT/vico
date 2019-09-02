echo "Downloading Imagenet urls ..."
URL="http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz"
OUT_DIR="${PWD}/symlinks/data/imagenet"
mkdir $OUT_DIR
wget --directory-prefix=$OUT_DIR $URL

echo "Uncompressing tgz file ..."
URL_TGZ_FILE="${OUT_DIR}/imagenet_fall11_urls.tgz"
tar -xvzf $URL_TGZ_FILE -C $OUT_DIR

echo "Downloading Wordnet ID to words map ..."
WORDS_TXT_URL="http://image-net.org/archive/words.txt"
wget --directory-prefix=$OUT_DIR $WORDS_TXT_URL

echo "Downloading wordnet hierarchy ..."
IS_A_TXT_URL="http://www.image-net.org/archive/wordnet.is_a.txt"
wget --directory-prefix=$OUT_DIR $IS_A_TXT_URL

echo "Preprocessing ..."
bash data/imagenet/preprocess.sh