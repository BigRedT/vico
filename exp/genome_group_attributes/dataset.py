import os
import copy
import itertools
from tqdm import tqdm
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
        self.group = 'material' # 'material' / 'color' / 'shape'


class GenomeAttributesDataset(Dataset):
    def __init__(self,const):
        super(GenomeAttributesDataset,self).__init__()
        self.const = copy.deepcopy(const)
        self.attrs, self.attr_to_idx = self.load_group_attributes()
        self.objs, self.obj_to_idx = self.load_object_synsets()
        self.annos = self.load_annos()
        self.object_id_to_image_id = io.load_json_object(
            self.const.object_id_to_image_id_json)
        self.object_ids = self.get_object_ids()
        self.transforms = transforms.Compose([
                transforms.RandomCrop(self.const.img_size),
                transforms.RandomHorizontalFlip()])
        print('Num regions: ',len(self))
    
    def load_object_synsets(self):
        objs = io.load_json_object(self.const.object_synset_freqs_json).keys()
        objs = sorted(list(objs)) + ['__UNK__']
        obj_to_idx = {o:i for i,o in enumerate(objs)}
        return objs, obj_to_idx

    def get_object_ids(self):
        object_ids = []
        for object_id in self.annos.keys():
            if object_id in self.object_id_to_image_id:
                object_ids.append(object_id)
        
        return sorted(object_ids)

    def load_group_attributes(self):
        attr_groups = io.load_json_object(self.const.attribute_groups_json)
        attrs = sorted(attr_groups[self.const.group])
        attr_to_idx = {a:i for i,a in enumerate(attrs)}
        return attrs, attr_to_idx

    def load_annos(self):
        annos_json = os.path.join(
            self.const.proc_dir,
            f'object_annos_{self.const.group}.json')
        annos = io.load_json_object(annos_json)
        return annos

    def create_label_vec(self,attributes):
        num_attributes = len(self.attrs)
        label = np.zeros([num_attributes],dtype=np.float32)
        pos_idxs = []
        for attr in attributes:
            idx = self.attr_to_idx[attr]
            label[idx] = 1.0
            pos_idxs.append(idx)
        return label, pos_idxs

    def __len__(self):
        return len(self.object_ids)

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

    def normalize(self,imgs,mean,std):
        b,h,w,c = imgs.shape
        imgs = (imgs-mean) / std
        return imgs

    def __getitem__(self,i):
        object_id = self.object_ids[i]
        image_id = self.object_id_to_image_id[object_id]
        anno = self.annos[object_id]
        attrs = anno['attribute_synsets']
        obj = anno['object_synset']
        obj_id = self.obj_to_idx[obj]
        box = anno['attribute_box']
        label_vec, pos_idxs = self.create_label_vec(attrs)
        img, img_path = self.get_image(image_id)
        try:
            region = self.crop_region(img,box,0)
            region = self.scale_smaller_side(region,self.const.img_size)
            region = self.transforms(region)
            region = np.array(region)
        except:
            return None

        to_return = {
            'region': region,
            'attributes': attrs,
            'object': obj,
            'object_id': obj_id,
            'label_vec': label_vec,
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
                if k=='attributes':
                    pass
                else:
                    batch_[k] = default_collate(batch_[k])

            return batch_

        return collate_fn


if __name__=='__main__':
    const = GenomeAttributesDatasetConstants()
    dataset = GenomeAttributesDataset(const)
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=10,
        collate_fn=dataset.create_collate_fn())
    count = 0
    for data in tqdm(dataloader):
        import pdb; pdb.set_trace()
        if len(data['attributes']) < 10:
            count+=1
    
    print(count)