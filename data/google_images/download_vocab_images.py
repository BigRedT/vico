import os
import nltk
from nltk.stem import WordNetLemmatizer
import argparse
from tqdm import tqdm
from google_images_download import google_images_download

import utils.io as io


parser = argparse.ArgumentParser()
parser.add_argument(
    '--vocab_json', 
    type=str, 
    default=None, 
    help='Path to vocab json file')
parser.add_argument(
    '--images_per_word',
    type=int,
    default=100,
    help='Number of images per word')
parser.add_argument(
    '--outdir', 
    type=str, 
    default=None, 
    help='Path where images will be downloaded')


def main():
    args = parser.parse_args()

    io.mkdir_if_not_exists(args.outdir)

    vocab = io.load_json_object(args.vocab_json)

    # Setup google downloaded
    response = google_images_download.googleimagesdownload()
    search_arguments = {
        'keywords': None,
        'limit': args.images_per_word,
        'safe_search': True,
        'output_directory': args.outdir,
        'usage_rights': 'labeled-for-noncommercial-reuse-with-modification'
    }

    # Download images for words
    num_words = len(vocab)
    for i, word in enumerate(vocab.keys()):
        print('-'*80)
        print(f'Downloading images for word: {word} ({i+1} / {num_words})')
        print('-'*80)
        search_arguments['keywords'] = word
        absolute_image_paths = response.download(search_arguments)

if __name__=='__main__':
    main()