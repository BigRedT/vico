import os
import glob
import nltk
from nltk.stem import WordNetLemmatizer
import argparse
from PIL import Image
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
    '--longer_size',
    type=int,
    default=224,
    help='Size of the longer dimension of image')
parser.add_argument(
    '--outdir', 
    type=str, 
    default=None, 
    help='Path where images will be downloaded')


def scale_and_pad(img,longer_size,pad=True,pad_value=128):
        h = longer_size
        w = longer_size
        img_w, img_h = img.size

        if img_h > img_w:
            img_w = int(img_w * h / float(img_h))
            img_h = h
        else:
            img_h = int(img_h * w / float(img_w))
            img_w = w

        img = img.resize((img_w,img_h))

        if pad is True:
            img = np.array(img)
            pad_w_left = (w - img_w)//2
            pad_w_right = w - pad_w_left - img_w
            pad_h_top = (h - img_h)//2
            pad_h_bottom = h - pad_h_top - img_h
            img = np.pad(
                img,
                ((pad_h_top,pad_h_bottom),(pad_w_left,pad_w_right),(0,0)),
                'constant',
                constant_values=pad_value)
            img = Image.fromarray(img)

        return img


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
        'usage_rights': 'labeled-for-noncommercial-reuse-with-modification',
        'color_type': 'full-color',
        'type': 'photo',
    }

    # Download images for words
    num_words = len(vocab)
    for i, word in enumerate(vocab.keys()):
        print('-'*80)
        print(f'Downloading images for word: {word} ({i+1} / {num_words})')
        print('-'*80)
        
        search_arguments['keywords'] = word
        
        # dirname = os.path.join(args.outdir,word)
        # if os.path.exists(dirname):
        #     continue
        
        absolute_image_paths = response.download(search_arguments)[word]
        for j,path in enumerate(absolute_image_paths):
            if not os.path.exists(path):
                continue
            
            try:
                img = Image.open(path)
                img = scale_and_pad(img,args.longer_size,pad=False)
                filename = os.path.join(args.outdir,f'{word}/{j}.png')
                img.save(filename)
            except:
                print(f'Could not read {path}')
            
            os.remove(path)


if __name__=='__main__':
    main()