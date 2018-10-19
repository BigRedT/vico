#!/bin/bash
OUTDIR="${PWD}/symlinks/data/google_images"
VOCAB_JSON="${PWD}/symlinks/data/glove/proc/glove_6B_300d_lemmatized_word_to_idx.json"
IMAGES_PER_WORD=20
python -m data.google_images.download_vocab_images \
    --vocab_json $VOCAB_JSON \
    --images_per_word  $IMAGES_PER_WORD \
    --outdir $OUTDIR