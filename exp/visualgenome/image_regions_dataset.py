import os
import copy
import itertools
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import utils.io as io
from torch.utils.data.dataloader import default_collate
from data.visualgenome.constants import VisualGenomeConstants

class ImageRegionsDatasetConstants(VisualGenomeConstants):
    def __init__(
            self,
            raw_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/visualgenome/raw'),
            proc_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/visualgenome/proc')):
        super(ImageRegionsDatasetConstants,self).__init__(raw_dir,proc_dir)
        self.crop_size = [224,224]
        

class ImageRegionsDataset(Dataset):
    def __init__(self,const):
        super(ImageRegionsDataset,self).__init__()
        self.const = const
        self.object_synset_freqs = io.load_json_object(
            self.const.object_synset_freqs_json)
        self.attribute_synset_freqs = io.load_json_object(
            self.const.attribute_synset_freqs_json)
        self.sorted_object_synsets = sorted(
            list(self.object_synset_freqs.keys()))
        self.sorted_attribute_synsets = sorted(
            list(self.attribute_synset_freqs.keys()))
        self.object_synset_to_idx = {
            k:v for v,k in enumerate(self.sorted_object_synsets)}
        self.attribute_synset_to_idx = {
            k:v for v,k in enumerate(self.sorted_attribute_synsets)}
        self.object_annos = io.load_json_object(self.const.object_annos_json)
        self.image_id_to_object_id = io.load_json_object(
            self.const.image_id_to_object_id_json)
        self.image_ids = list(self.image_id_to_object_id.keys())

    def get_image(self,image_id):
        filename = os.path.join(self.const.img_dir1,f'{image_id}.jpg')
        if not os.path.exists(filename):
            filename = os.path.join(self.const.img_dir2,f'{image_id}.jpg')
        img = Image.open(filename)
        return img

    def crop_region(self,img,bbox,context_percent=10):
        x,y,w,h = bbox
        cw = context_percent*w*0.01
        ch = context_percent*h*0.01
        return img.crop(box=(x-cw,y-ch,x+w+cw,y+h+ch))

    def scale_and_pad(self,img,size):
        h,w = size
        img_w, img_h = img.size
        if img_h > img_w:
            img_w = int(img_w * h / float(img_h))
            img_h = h
        else:
            img_h = int(img_h * w / float(img_w))
            img_w = w
        img = img.resize((img_w,img_h))
        img = np.array(img).astype(np.float32)
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

    def create_object_label(self,object_synsets):
        num_objects = len(self.sorted_object_synsets)
        label = np.zeros([num_objects])
        idx = -1
        for synset in object_synsets:
            #if synset in self.object_synset_to_idx:
            idx = self.object_synset_to_idx[synset]
            label[idx] = 1.0
            # else:
            #     import pdb; pdb.set_trace()
        return label, idx

    def create_attribute_label(self,attribute_synsets):
        num_attributes = len(self.sorted_attribute_synsets)
        label = np.zeros([num_attributes])
        for synset in attribute_synsets:
            if synset in self.attribute_synset_to_idx:
                idx = self.attribute_synset_to_idx[synset]
                label[idx] = 1.0
        return label

    def normalize(self,imgs,mean,std):
        b,h,w,c = imgs.shape
        imgs = (imgs-mean) / std
        return imgs

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,i):
        image_id = self.image_ids[i]
        object_ids = self.image_id_to_object_id[image_id]
        num_regions = len(object_ids)
        if num_regions==0:
            return None

        img = self.get_image(image_id)
        img_w,img_h = img.size
        regions = []
        object_synsets = []
        attribute_synsets = []
        object_labels = []
        attribute_labels = []
        object_labels_idx = []
        boxes = []
        img_size = []
        for object_id in object_ids:
            object_anno = self.object_annos[object_id]
            boxes.append(object_anno['object_box'])
            try:
                region = self.crop_region(img,object_anno['object_box'],0)
                region = self.scale_and_pad(region,self.const.crop_size)
                regions.append(region)
            except:
                return None
            object_synsets.append(object_anno['object_synsets'])
            attribute_synsets.append(
                object_anno['attribute_synsets'])
            object_label, object_label_idx = self.create_object_label(
                object_anno['object_synsets'])
            object_labels.append(object_label)
            object_labels_idx.append(object_label_idx)
            attribute_labels.append(
                self.create_attribute_label(object_anno['attribute_synsets']))
        to_return = {
            #'image': [np.array(img).astype(np.float32)],
            'image_ids': [image_id]*num_regions,
            'object_ids': object_ids,
            'regions': regions,
            'boxes': boxes, # coordinates in int
            'image_sizes': [[img_h,img_w]]*num_regions, # dimensions in int
            'object_synsets': object_synsets,
            'attribute_synsets': attribute_synsets,
            'object_labels': object_labels,
            'object_labels_idx': object_labels_idx,
            'attribute_labels': attribute_labels,
            'object_labels_idx': object_labels_idx
        }
        return to_return

    def create_collate_fn(self,non_tensor_keys):
        def collate_fn(batch):
            batch = [sample for sample in batch if sample is not None]
            if len(batch) == 0:
                return None
        
            batch_ = {}
            for k in batch[0].keys():
                batch_[k] = []
                for sample in batch:
                    batch_[k] = batch_[k] + sample[k]
                if k in non_tensor_keys:
                    continue
                else:
                    try:
                        batch_[k] = np.stack(batch_[k])
                    except:
                        import pdb; pdb.set_trace()
            return batch_
        return collate_fn


if __name__=='__main__':
    const = ImageRegionsDatasetConstants()
    dataset = ImageRegionsDataset(const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets'])
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=5,
        collate_fn=collate_fn)
    for data in dataloader:
        print('Voila')
        import pdb; pdb.set_trace()
    print('done')
