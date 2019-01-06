import os
import h5py
import math
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from tensorboard_logger import configure, log_value
import numpy as np

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
import utils.pytorch_layers as pytorch_layers
from .models.resnet import ResnetModel
from .models.embed_to_classifier import Embed2Class
from .dataset import Cifar100Dataset


def train_model(model,dataloaders,exp_const):
    model.embed2class.load_embeddings(dataloaders['training'].dataset.labels)
    model.embed2class.embed.weight.requires_grad=False
    embed2class_params = []
    for name,p in model.embed2class.named_parameters():
        if 'embed' not in name:
            embed2class_params.append(p)
        else:
            print(f'Not optimizing {name}')
    
    params = itertools.chain(
        model.net.parameters(),
        embed2class_params)
    
    lr = exp_const.lr
    if exp_const.optimizer == 'SGD':
        opt = optim.SGD(
            params,
            lr=lr,
            momentum=exp_const.momentum,
            weight_decay=1e-4)
    elif exp_const.optimizer == 'Adam':
        opt = optim.Adam(
            params,
            lr=lr)
    elif exp_const.optimizer == 'Adagrad':
        opt = optim.Adagrad(
            params,
            lr=lr)
    else:
        assert(False), 'optimizer not implemented'

    criterion = nn.CrossEntropyLoss()

    step = 0
    if model.const.model_num is not None:
        step = model.const.model_num

    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean))
    img_std = Variable(torch.cuda.FloatTensor(model.img_std))

    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloaders['training']):
            # Set mode
            model.net.train()
            model.embed2class.train()

            # Forward pass
            imgs = Variable(data['img'].cuda().float()/255)
            imgs = dataloaders['training'].dataset.normalize(
                imgs,
                img_mean,
                img_std)
            imgs = imgs.permute(0,3,1,2)
            label_idxs = Variable(data['label_idx'].cuda())

            if exp_const.feedforward==True:
                logits, feats = model.net(imgs)
            else:
                _, feats = model.net(imgs)
                class_weights = model.embed2class()
                logits = model.embed2class.classify(feats,class_weights)

            # Computer loss
            loss = criterion(logits,label_idxs)
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                _,argmax = torch.max(logits,1)
                argmax = argmax.data.cpu().numpy()
                label_idxs_ = label_idxs.data.cpu().numpy()
                train_acc = np.mean(argmax==label_idxs_)

                log_items = {
                    'Loss': loss.data[0],
                    'Train Acc': train_acc,
                }

                log_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                for name,value in log_items.items():
                    log_str += '{}: {:.4f} | '.format(name,value)
                    log_value(name,value,step)

                print(log_str)
            
            if step%(10*exp_const.log_step)==0:
                print(f'Experiment: {exp_const.exp_name}')
                
            if step%exp_const.model_save_step==0:
                save_items = {
                    'net': model.net,
                    'embed2class': model.embed2class
                }

                for name,nn_model in save_items.items():
                    model_path = os.path.join(
                        exp_const.model_dir,
                        f'{name}_{step}')
                    torch.save(nn_model.state_dict(),model_path)

            if step%exp_const.val_step==0:
                eval_results = eval_model(
                    model,
                    dataloaders['test'],
                    exp_const,
                    step)
                print(eval_results)
                log_value('Test Acc',eval_results['Acc'],step)
            
            if step==32000:
                lr = 0.1*lr
                pytorch_layers.set_learning_rate(opt,lr)
                
            step += 1


def eval_model(model,dataloader,exp_const,step):
    # Set mode
    model.net.eval()
    model.embed2class.eval()

    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean))
    img_std = Variable(torch.cuda.FloatTensor(model.img_std))

    criterion = nn.CrossEntropyLoss()

    avg_loss = 0
    correct = 0
    num_samples = 0
    for it,data in enumerate(tqdm(dataloader)):
        # if num_samples >= exp_const.num_val_samples:
        #     break

        # Forward pass
        imgs = Variable(data['img'].cuda().float()/255)
        imgs = dataloader.dataset.normalize(
            imgs,
            img_mean,
            img_std)
        imgs = imgs.permute(0,3,1,2)
        label_idxs = Variable(data['label_idx'].cuda())
        if exp_const.feedforward==True:
            logits,feats = model.net(imgs)
        else:
            _, feats = model.net(imgs)
            class_weights = model.embed2class()
            logits = model.embed2class.classify(feats,class_weights)

        # Computer loss
        loss = criterion(logits,label_idxs)    

        _,argmax = torch.max(logits,1)
        argmax = argmax.data.cpu().numpy()
        label_idxs_ = label_idxs.data.cpu().numpy()

        # Aggregate loss or accuracy
        batch_size = imgs.size(0)
        num_samples += batch_size
        avg_loss += (loss.data[0]*batch_size)
        correct += np.sum(argmax==label_idxs_)

    avg_loss = avg_loss / num_samples
    acc = correct / float(num_samples)

    eval_results = {
        'Avg Loss': avg_loss, 
        'Acc': acc,
    }

    return eval_results


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    io.mkdir_if_not_exists(exp_const.vis_dir)
    configure(exp_const.log_dir)
    save_constants(
        {'exp': exp_const,'data': data_const,'model': model_const},
        exp_const.exp_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.net = ResnetModel(model.const.net)
    model.embed2class = Embed2Class(model.const.embed2class)
    if model.const.model_num is not None:
        model.net.load_state_dict(torch.load(model.const.net_path))
        model.embed2class.load_state_dict(
            torch.load(model.const.embed2class_path))
    model.net.cuda()
    model.embed2class.cuda()
    model.img_mean = np.array([0.485, 0.456, 0.406])
    model.img_std = np.array([0.229, 0.224, 0.225])
    model.to_file(os.path.join(exp_const.exp_dir,'model.txt'))

    print('Creating dataloader ...')
    dataloaders = {}
    for mode, subset in exp_const.subset.items():
        data_const = copy.deepcopy(data_const)
        if subset=='train':
            data_const.train = True
        else:
            data_const.train = False
        dataset = Cifar100Dataset(data_const)
        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=exp_const.batch_size,
            shuffle=True,
            num_workers=exp_const.num_workers)

    train_model(model,dataloaders,exp_const)