# No need to run this if using torchvision

OUT_DIR="${PWD}/symlinks/data/cifar100"
mkdir $OUT_DIR

echo "Downloading CIFAR 100 ..."
URL="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
wget --directory-prefix=$OUT_DIR $URL

echo "Extract CIFAR 100 ..."
TAR_GZ_FILE="${OUT_DIR}/cifar-100-python.tar.gz"
tar xvzf $TAR_GZ_FILE -C $OUT_DIR