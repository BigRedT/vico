import os
import copy
import glob
import itertools
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils.io as io
from torch.utils.data.dataloader import default_collate
from data.google_images.constants import GoogleImagesConstants


class GoogleImagesImageLevelDatasetConstants(GoogleImagesConstants):
    def __init__(
            self,
            raw_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/google_images')):
        super(GoogleImagesImageLevelDatasetConstants,self).__init__(raw_dir=raw_dir)
        self.vocab_json = None
        self.longer_size = 224


class GoogleImagesImageLevelDataset(Dataset):
    def __init__(self,const):
        super(GoogleImagesImageLevelDataset,self).__init__()
        self.const = copy.deepcopy(const)
        self.vocab = self.read_vocab(self.const.vocab_json)
        self.paths, self.words = self.read_paths_and_words()
        self.word_to_idx, self.idx_to_word = self.create_word_idx()

    def read_vocab(self,vocab_json):
        assert(not (vocab_json is None)), 'Provide vocab_json'
        return io.load_json_object(vocab_json)
    
    def create_word_idx(self):
        idx_to_word = sorted(list(self.vocab.keys()))
        word_to_idx = {w:i for i,w in enumerate(idx_to_word)}
        return word_to_idx, idx_to_word

    def read_paths_and_words(self):
        paths = []
        words = []
        for word in self.vocab.keys():
            dirname = os.path.join(self.const.raw_dir,word)
            img_paths = glob.glob(f'{dirname}/*.png')
            paths += img_paths
            words += [word]*len(img_paths)
        return paths, words

    def __len__(self):
        return len(self.paths)

    def scale_and_pad(self,img,longer_size,scale=True,pad=True):
        h = w = longer_size
        img_w, img_h = img.size
        if img_h > img_w:
            img_w = int(img_w * h / float(img_h))
            img_h = h
        else:
            img_h = int(img_h * w / float(img_w))
            img_w = w
        
        if scale is True:
            img = img.resize((img_w,img_h))
        
        img = np.array(img).astype(np.float32)

        # Convert single channel image to 3 channel
        if len(img.shape)==2:
            img = np.tile(img[:,:,np.newaxis],[1,1,3])
        
        # Convert 4 channel image to 3 channel by dropping transparency
        if img.shape[2]==4:
            img = img[:,:,:3]

        if pad is True:
            pad_w_left = (w - img_w)//2
            pad_w_right = w - pad_w_left - img_w
            pad_h_top = (h - img_h)//2
            pad_h_bottom = h - pad_h_top - img_h
            img = np.pad(
                img,
                ((pad_h_top,pad_h_bottom),(pad_w_left,pad_w_right),(0,0)),
                'constant',
                constant_values=0.449*255)

        return img

    def normalize(self,imgs,mean,std):
        b,h,w,c = imgs.shape
        imgs = (imgs-mean) / std
        return imgs

    def __getitem__(self,i):
        try:
            img = Image.open(self.paths[i])
        except:
            return None

        if max(img.size)==self.const.longer_size:
            scale = False
        else:
            scale = True
        
        img = self.scale_and_pad(
            img,
            self.const.longer_size,
            scale=scale,
            pad=True)
        
        word = self.words[i]
        to_return = {
            'path': self.paths[i],
            'img': img,
            'word': word,
            'idx': self.word_to_idx[word],
        }
        
        return to_return

    def create_collate_fn(self):
        def collate_fn(batch):
            batch = [sample for sample in batch if sample is not None]
            if len(batch)==0:
                return None
            
            batch_ = {}
            for k in batch[0].keys():
                batch_[k] = default_collate([sample[k] for sample in batch])

            return batch_
        
        return collate_fn


if __name__=='__main__':
    const = GoogleImagesImageLevelDatasetConstants()
    const.vocab_json = os.path.join(
        os.getcwd(),
        'symlinks/data/semeval_2018_10/proc/all_words.json')
    dataset = GoogleImagesImageLevelDataset(const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        collate_fn=collate_fn)
    for data in dataloader:
        import pdb; pdb.set_trace()

