import os
import h5py
import math
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

import utils.io as io
from utils.constants import save_constants
from exp.visualgenome.image_regions_dataset import ImageRegionsDataset
from exp.visualgenome.models.resnet import resnet152


def estimate_num_objects(image_id_to_object_id):
    num_objects = 0
    for image_id, object_ids in image_id_to_object_id.items():
        num_objects += len(object_ids)
    return num_objects


def main(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir)
    save_constants({'exp': exp_const,'data': data_const},exp_const.exp_dir)
    
    print('Creating network ...')
    net = resnet152(pretrained=True).cuda()
    net.eval()
    imagenet_labels = io.load_json_object(data_const.imagenet_labels_json)

    print('Creating dataloader ...')
    dataset = ImageRegionsDataset(data_const)
    collate_fn = dataset.create_collate_fn(
        ['object_synsets','attribute_synsets','object_ids'])
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=False,
        num_workers=exp_const.num_workers,
        collate_fn=collate_fn)

    print('Creating visual_features.h5py ...')
    features_h5py = h5py.File(
        os.path.join(exp_const.exp_dir,'visual_features.h5py'),
        'w')

    num_objects = estimate_num_objects(dataset.image_id_to_object_id)
    features = features_h5py.create_dataset(
        'features',
        (num_objects,exp_const.feature_dim))

    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])

    object_id_to_features_idx = {}
    object_id_to_pred_labels = {}
    object_count = 0
    for data in tqdm(dataloader):
        # if object_count > 100:
        #     break
        if data is None:
            continue
    
        regions = dataloader.dataset.normalize(
            data['regions']/255,
            img_mean,
            img_std)
        regions = Variable(torch.FloatTensor(regions).cuda(),volatile=True)
        regions = regions.permute(0,3,1,2)
        B = regions.size(0)
        out = []
        last_layer_features = []
        for i in range(math.ceil(B/256)):
            r = min(i*256+256,B)
            try:
                out_, last_layer_features_ = net(regions[i*256:r])
            except: 
                import pdb; pdb.set_trace()
            out.append(out_)
            last_layer_features.append(last_layer_features_)
        out = torch.cat(out,0)
        last_layer_features = torch.cat(last_layer_features,0)
        last_layer_features = last_layer_features.data.cpu().numpy()
        label_idxs = torch.topk(out,10,1)
        label_idxs = label_idxs[1].data.cpu().numpy()
        
        for j in range(B):
            object_id = data['object_ids'][j]
            labels = []
            for k in range(5):
                idx = label_idxs[j][k]
                labels.append([idx,imagenet_labels[idx]])
            object_id_to_pred_labels[object_id] = labels
            object_id_to_features_idx[object_id] = object_count
            features[object_count] = last_layer_features[j]
            object_count += 1
    
    features_h5py.close()
    object_id_to_pred_labels_json = os.path.join(
        exp_const.exp_dir,
        'object_id_to_pred_labels.json')
    io.dump_json_object(object_id_to_pred_labels,object_id_to_pred_labels_json)
    object_id_to_features_idx_json = os.path.join(
        exp_const.exp_dir,
        'object_id_to_features_idx.json')
    io.dump_json_object(
        object_id_to_features_idx,
        object_id_to_features_idx_json)
    