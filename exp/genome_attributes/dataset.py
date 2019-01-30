import os
import copy
import itertools
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

import utils.io as io
from data.visualgenome.constants import VisualGenomeConstants

class GenomeAttributesDatasetConstants(VisualGenomeConstants):
    def __init__(
            self,
            raw_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/visualgenome/raw'),
            proc_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/visualgenome/proc')):
        super(GenomeAttributesDatasetConstants,self).__init__(raw_dir,proc_dir)
        self.img_size = 224
        

class GenomeAttributesDataset(Dataset):
    def __init__(self,const):
        super(GenomeAttributesDataset,self).__init__()
        self.const = copy.deepcopy(const)
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
        self.transforms = transforms.Compose(
            [
                transforms.RandomCrop(self.const.img_size),
                transforms.RandomHorizontalFlip(),
            ]
        )

    def get_image(self,image_id):
        filename = os.path.join(self.const.img_dir1,f'{image_id}.jpg')
        if not os.path.exists(filename):
            filename = os.path.join(self.const.img_dir2,f'{image_id}.jpg')
        img = Image.open(filename).convert('RGB')
        return img, filename

    def crop_region(self,img,bbox,context_percent=10):
        x,y,w,h = bbox
        cw = context_percent*w*0.01
        ch = context_percent*h*0.01
        return img.crop(box=(x-cw,y-ch,x+w+cw,y+h+ch))

    def scale_smaller_side(self,img,smaller_size):
        img_w, img_h = img.size
        if img_h < img_w:
            img_w = int(img_w * smaller_size / float(img_h))
            img_h = smaller_size
        else:
            img_h = int(img_h * smaller_size / float(img_w))
            img_w = smaller_size
        
        img = img.resize((img_w,img_h))

        return img

    def create_object_label(self,object_synsets):
        num_objects = len(self.sorted_object_synsets)
        label = np.zeros([num_objects],dtype=np.float32)
        idx = -1
        for synset in object_synsets:
            idx = self.object_synset_to_idx[synset]
            label[idx] = 1.0
        return label, idx

    def create_attribute_label(self,attribute_synsets):
        num_attributes = len(self.sorted_attribute_synsets)
        label = np.zeros([num_attributes],dtype=np.float32)
        pos_idxs = []
        for synset in attribute_synsets:
            idx = self.attribute_synset_to_idx[synset]
            label[idx] = 1.0
            pos_idxs.append(idx)
        return label, pos_idxs

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

        img, img_path = self.get_image(image_id)
        img_w,img_h = img.size
        regions = []
        object_synsets = []
        object_words = []
        attribute_synsets = []
        attribute_words = []
        object_labels = []
        attribute_labels = []
        object_labels_idx = []
        attribute_labels_idxs = []
        boxes = []
        img_size = []
        object_ids_ = []
        for object_id in object_ids:
            object_anno = self.object_annos[object_id]
            
            if len(object_anno['attribute_synsets'])==0:
                continue
            
            boxes.append(object_anno['attribute_box'])
            
            try:
                region = self.crop_region(img,object_anno['attribute_box'],0)
                region = self.scale_smaller_side(region,self.const.img_size)
                region = self.transforms(region)
                regions.append(region)
            except:
                return None
            object_words.append(object_anno['names'])
            object_synsets.append(object_anno['object_synsets'])
            object_label, object_label_idx = self.create_object_label(
                object_anno['object_synsets'])
            object_labels.append(object_label)
            object_labels_idx.append(object_label_idx)
            
            attribute_words.append(object_anno['attributes'])
            attribute_synsets.append(object_anno['attribute_synsets'])
            attribute_label, attribute_pos_idxs = self.create_attribute_label(
                object_anno['attribute_synsets'])
            attribute_labels.append(attribute_label)
            attribute_labels_idxs.append(attribute_pos_idxs)

            object_ids_.append(object_id)
        
        to_return = {
            #'image': [np.array(img).astype(np.float32)],
            'image_ids': [image_id]*num_regions,
            'object_ids': object_ids_,
            'regions': regions,
            'boxes': boxes, # coordinates in int
            'image_sizes': [[img_h,img_w]]*num_regions, # dimensions in int
            'object_synsets': object_synsets,
            'object_words': object_words,
            'attribute_synsets': attribute_synsets,
            'attribute_words': attribute_words,
            'object_labels': object_labels,
            'object_labels_idx': object_labels_idx,
            'attribute_labels': attribute_labels,
            'attribute_labels_idxs': attribute_labels_idxs,
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
                        batch_[k] = default_collate(np.stack(batch_[k]))
                    except:
                        import pdb; pdb.set_trace()
            return batch_
        return collate_fn


class GenomeAttributesNoImgsDataset(GenomeAttributesDataset):
    def __getitem__(self,i):
        image_id = self.image_ids[i]
        object_ids = self.image_id_to_object_id[image_id]
        num_regions = len(object_ids)
        if num_regions==0:
            return None

        regions = []
        object_synsets = []
        object_words = []
        attribute_synsets = []
        attribute_words = []
        object_labels = []
        attribute_labels = []
        object_labels_idx = []
        attribute_labels_idxs = []
        boxes = []
        img_size = []
        object_ids_ = []
        for object_id in object_ids:
            object_anno = self.object_annos[object_id]
            
            if len(object_anno['attribute_synsets'])==0:
                continue
            
            boxes.append(object_anno['attribute_box'])
            
            object_words.append(object_anno['names'])
            object_synsets.append(object_anno['object_synsets'])
            object_label, object_label_idx = self.create_object_label(
                object_anno['object_synsets'])
            object_labels.append(object_label)
            object_labels_idx.append(object_label_idx)
            
            attribute_words.append(object_anno['attributes'])
            attribute_synsets.append(object_anno['attribute_synsets'])
            attribute_label, attribute_pos_idxs = self.create_attribute_label(
                object_anno['attribute_synsets'])
            attribute_labels.append(attribute_label)
            attribute_labels_idxs.append(attribute_pos_idxs)

            object_ids_.append(object_id)
        
        to_return = {
            #'image': [np.array(img).astype(np.float32)],
            'image_ids': [image_id]*num_regions,
            'object_ids': object_ids_,
            'boxes': boxes, # coordinates in int
            #'image_sizes': [[img_h,img_w]]*num_regions, # dimensions in int
            'object_words': object_words,
            'object_synsets': object_synsets,
            'attribute_words': attribute_words,
            'attribute_synsets': attribute_synsets,
            'object_labels': object_labels,
            'object_labels_idx': object_labels_idx,
            'attribute_labels': attribute_labels,
            'attribute_labels_idxs': attribute_labels_idxs,
        }
        return to_return


if __name__=='__main__':
    const = GenomeAttributesDatasetConstants()
    dataset = GenomeAttributesDataset(const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets','attribute_labels_idxs'])
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=collate_fn)
    for data in dataloader:
        print('Voila')
        import pdb; pdb.set_trace()
    print('done')
