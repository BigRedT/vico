# Copy this file in the pythia directory and execute as a bash script

mkdir data
cd data
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/vqa2.0_glove.6B.300d.txt.npy
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/vocabulary_vqa.txt
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/answers_vqa.txt
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/imdb.tar.gz
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/rcnn_10_100.tar.gz
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/detectron.tar.gz
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/large_vocabulary_vqa.txt
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/large_vqa2.0_glove.6B.300d.txt.npy
gunzip imdb.tar.gz 
tar -xf imdb.tar

gunzip rcnn_10_100.tar.gz 
tar -xf rcnn_10_100.tar
rm -f rcnn_10_100.tar

gunzip detectron.tar.gz
tar -xf detectron.tar
rm -f detectron.tar