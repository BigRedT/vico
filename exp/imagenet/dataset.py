import os
import copy
import itertools
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

import utils.io as io
from data.imagenet.constants import ImagenetConstants
from data.imagenet.wordnet import WordNet, WordNetConstants


class ImagenetDatasetConstants(ImagenetConstants):
    def __init__(
            self,
            raw_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/imagenet')):
        super(ImagenetDatasetConstants,self).__init__(raw_dir)
        self.img_size = 224


class ImagenetDataset(Dataset):
    def __init__(self,const):
        super(ImagenetDataset,self).__init__()
        self.const = copy.deepcopy(const)
        self.wordnet = self.create_wordnet()
        self.wnids, self.wnid_to_idx = self.get_wnids()
        self.wnid_and_img_paths = self.get_wnid_and_img_paths()
        self.transforms = transforms.Compose(
            [
                transforms.RandomCrop(self.const.img_size),
                transforms.RandomHorizontalFlip(),
            ]
        )

    def get_wnid_and_img_paths(self):
        print('Reading image paths ...')
        wnid_and_img_paths = []
        for wnid in tqdm(self.wnids):
            wnid_dir = os.path.join(self.const.img_dir,wnid)
            if not os.path.exists(wnid_dir):
                continue
            
            root, _, fnames = list(os.walk(wnid_dir))[0]
            for fname in sorted(fnames):
                if not (fname.lower().endswith('.jpg') or 
                        fname.lower().endswith('.png')):
                    continue
                path = os.path.join(root, fname)
                wnid_and_img_paths.append((wnid,path))
        
        return wnid_and_img_paths

    def get_image(self,img_path):
        img = Image.open(img_path).convert('RGB')
        return img

    def create_wordnet(self):
        print('Creating WordNet instance ...')
        wordnet = WordNet(self.const)
        return wordnet

    def get_wnids(self):
        wnids = sorted(list(self.wordnet.nodes.keys()))
        wnid_to_idx = {wnid:i for i,wnid in enumerate(wnids)}
        return wnids, wnid_to_idx

    def get_on_wnids(self,wnid):
        ancestors = self.wordnet.get_ancestors(self.wordnet.nodes[wnid])
        on_wnids = [wnid]
        for node in ancestors:
            on_wnids.append(node.wnid)
        return on_wnids

    def get_unknown_wnids(self,wnid):
        subtree = self.wordnet.get_nodes_in_subtree(self.wordnet.nodes[wnid])
        unknown_wnids = [node.wnid for node in subtree]
        return unknown_wnids

    def get_label_vec(self,wnid):
        label_vec = np.zeros([len(self.wordnet.nodes)],dtype=np.uint8)
        on_wnids = self.get_on_wnids(wnid)
        for wnid in on_wnids:
            i = self.wnid_to_idx[wnid]
            label_vec[i] = 1
        return label_vec, on_wnids

    def get_weight_vec(self,wnid):
        weight_vec = np.ones([len(self.wordnet.nodes)],dtype=np.uint8)
        unknown_wnids = self.get_unknown_wnids(wnid)
        for wnid in unknown_wnids:
            i = self.wnid_to_idx[wnid]
            weight_vec[i] = 0
        return weight_vec

    def normalize(self,imgs,mean,std):
        b,h,w,c = imgs.shape
        imgs = (imgs-mean) / std
        return imgs

    def __len__(self):
        return len(self.wnid_and_img_paths)

    def __getitem__(self,i):
        wnid, img_path = self.wnid_and_img_paths[i]
        label_vec, on_wnids = self.get_label_vec(wnid)
        weight_vec = self.get_weight_vec(wnid)
        try:
            img = self.get_image(img_path)
            img = self.transforms(img)
        except:
            return None
        
        to_return = {
            'wnid': wnid,
            'img_path': img_path,
            'label_vec': label_vec,
            'weight_vec': weight_vec,
            'img': np.array(img),
            'on_wnids': on_wnids,
        }
        return to_return

    def create_collate_fn(self):
        def collate_fn(batch):
            batch = [sample for sample in batch if sample is not None]
            if len(batch) == 0:
                return None
        
            batch_ = {}
            for k in batch[0].keys():
                batch_[k] = [sample[k] for sample in batch]
                if k=='on_wnids':
                    pass
                else:
                    batch_[k] = default_collate(batch_[k])

            return batch_

        return collate_fn


class ImagenetNoImgsDataset(ImagenetDataset):
    def get_wnid_and_img_paths(self):
        wnid_to_urls = io.load_json_object(self.const.wnid_to_urls_json)
        wnid_and_img_paths = []
        for wnid,urls in wnid_to_urls.items():
            for img_id,url in urls.items():
                wnid_and_img_paths.append((wnid,url))
        
        return wnid_and_img_paths

    def __getitem__(self,i):
        wnid, img_path = self.wnid_and_img_paths[i]
        label_vec, on_wnids = self.get_label_vec(wnid)
        weight_vec = self.get_weight_vec(wnid)
        to_return = {
            'wnid': wnid,
            'img_path': img_path,
            'label_vec': label_vec,
            'weight_vec': weight_vec,
            'on_wnids': on_wnids,
        }
        return to_return


if __name__=='__main__':
    const = ImagenetDatasetConstants()
    dataset = ImagenetNoImgsDataset(const)
    collate_fn = dataset.create_collate_fn()
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=5)
    for data in tqdm(dataloader):
        import pdb; pdb.set_trace()