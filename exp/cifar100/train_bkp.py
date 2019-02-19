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
from .models.conse import Conse
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
            weight_decay=0*1e-4)
    elif exp_const.optimizer == 'Adam':
        opt = optim.Adam(
            params,
            lr=lr,
            weight_decay=0*1e-4)
    elif exp_const.optimizer == 'Adagrad':
        opt = optim.Adagrad(
            params,
            lr=lr)
    else:
        assert(False), 'optimizer not implemented'

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    multimargin = nn.MultiMarginLoss(margin=0.2)

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
                reverse_loss = Variable(torch.FloatTensor([0])).cuda()
                #margin_loss = torch.Variable(torch.FloatTensor([0])).cuda()
            else:
                _, feats = model.net(imgs)
                class_weights = model.embed2class()
                logits = model.embed2class.classify(feats,class_weights)
                reverse_loss = Variable(torch.FloatTensor([0])).cuda()
                reverse_loss = model.embed2class.reverse_loss(class_weights)
                margin_loss = model.embed2class.margin_loss(logits,label_idxs)
                #margin_loss = multimargin(logits,label_idxs)

            # Computer loss
            loss = criterion(logits,label_idxs) + reverse_loss #+ margin_loss
            #loss = margin_loss

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
                    'Reverse Loss': reverse_loss.data[0]
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
                log_value('Test Super Acc',eval_results['Super Acc'],step)
                log_value('Seen Acc',eval_results['Seen Acc'],step)
                log_value('Unseen Acc',eval_results['Unseen Acc'],step)
                log_value('Seen Nll',eval_results['Seen Nll'],step)
                log_value('Unseen Nll',eval_results['Unseen Nll'],step)
                log_value('Zsl Acc',eval_results['Zsl Acc'],step)
            
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
    softmax = nn.Softmax(dim=1)
    multimargin = nn.MultiMarginLoss(margin=0.2)

    conse = Conse(
        10,
        model.embed2class.embed.weight,
        dataloader.dataset.held_out_idx)

    avg_loss = 0
    correct = 0
    correct_class = {l: 0 for l in dataloader.dataset.labels}
    sample_class = {l: 0 for l in dataloader.dataset.labels}
    nll_class = {l: 0 for l in dataloader.dataset.labels}
    super_correct = 0
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
        #loss = multimargin(logits,label_idxs)
        
        label_idxs_ = label_idxs.data.cpu().numpy()
        #prob = 0.5*(logits + 1)
        prob = softmax(logits)
        prob = prob.data.cpu().numpy()
        for i in range(prob.shape[1]):
            if i not in dataloader.dataset.held_out_idx:
                prob[:,i] = 0
        #prob = conse.infer_prob2(prob)
        #prob = prob.data.cpu().numpy()
        argmax = np.argmax(prob,1)
        #argmax = argmax.data.cpu().numpy()
        label_idxs_ = label_idxs.data.cpu().numpy()
        super_labels = data['super_label']
        gt_labels = data['label']
        for i in range(prob.shape[0]):
            pred_label = dataloader.dataset.labels[argmax[i]]
            gt_label = gt_labels[i]
            nll_class[gt_label] += -1*np.log(prob[i,label_idxs_[i]]+1e-3)
            sample_class[gt_label] += 1
            if gt_label==pred_label:
                correct_class[gt_label] += 1
            
            if super_labels[i]==dataloader.dataset.fine_to_super[pred_label]:
                super_correct += 1

        # Aggregate loss or accuracy
        batch_size = imgs.size(0)
        num_samples += batch_size
        avg_loss += (loss.data[0]*batch_size)
        correct += np.sum(argmax==label_idxs_)
    
    avg_loss = avg_loss / num_samples
    acc = correct / float(num_samples)
    super_acc = super_correct / float(num_samples)
    
    seen_acc = 0
    unseen_acc = 0
    seen_nll = 0
    unseen_nll = 0
    num_seen_classes = 0
    num_unseen_classes = 0
    for l in correct_class.keys():
        if l in dataloader.dataset.held_out_labels:
            unseen_nll += (nll_class[l] / sample_class[l])
            unseen_acc += (correct_class[l] / sample_class[l])
            num_unseen_classes += 1
        else:
            seen_nll += (nll_class[l] / sample_class[l])
            seen_acc += (correct_class[l] / sample_class[l])
            num_seen_classes += 1

    seen_nll = round(seen_nll / num_seen_classes,4)
    unseen_nll = round(unseen_nll / num_unseen_classes,4)
    seen_acc = round(seen_acc / num_seen_classes,4)
    unseen_acc = round(unseen_acc / num_unseen_classes,4)
    zsl_acc = 2*seen_acc*unseen_acc / (seen_acc+unseen_acc)

    eval_results = {
        'Avg Loss': avg_loss, 
        'Acc': acc,
        'Super Acc': super_acc,
        'Seen Acc': seen_acc,
        'Unseen Acc': unseen_acc,
        'Seen Nll': seen_nll,
        'Unseen Nll': unseen_nll,
        'Zsl Acc': zsl_acc,
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