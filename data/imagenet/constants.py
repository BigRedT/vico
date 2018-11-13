import os

import utils.io as io


class ImagenetConstants(io.JsonSerializableClass):
    def __init__(
            self,
            raw_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/imagenet')):
        self.raw_dir = raw_dir
        
        self.urls_txt = os.path.join(
            self.raw_dir,
            'fall11_urls.txt')
        self.words_txt = os.path.join(
            self.raw_dir,
            'words.txt')
        self.is_a_txt = os.path.join(
            self.raw_dir,
            'wordnet.is_a.txt')

        self.wnid_to_urls_json = os.path.join(
            self.raw_dir,
            'wnid_to_urls.json')
        self.wnid_to_words_json = os.path.join(
            self.raw_dir,
            'wnid_to_words.json')
        self.wnid_to_parent_json = os.path.join(
            self.raw_dir,
            'wnid_to_parent.json')

        self.img_dir = os.path.join(
            self.raw_dir,
            'images')