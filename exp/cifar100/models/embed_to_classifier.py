import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class Embed2ClassConstants(io.JsonSerializableClass):
    def __init__(self):
        super(Embed2ClassConstants,self).__init__()
        self.num_classes = 100
        self.glove_dim = 300
        self.embed_dims = 300
        self.embed_h5py = None
        self.embed_word_to_idx_json = None
        self.weights_dim = 64
        self.linear = True
        self.no_glove = False


class Embed2Class(nn.Module):
    def __init__(self,const):
        super(Embed2Class,self).__init__()
        self.const = copy.deepcopy(const)
        self.embed = nn.Embedding(
            self.const.num_classes,
            self.const.embed_dims)
        if self.const.linear==True:
            self.fc = nn.Linear(
                self.const.embed_dims,
                self.const.weights_dim,
                bias=False)
            self.reverse_fc = nn.Linear(
                self.const.weights_dim,
                self.const.embed_dims,
                bias=False)
        else:
            self.fc1 = nn.Linear(self.const.embed_dims,self.const.weights_dim)
            self.bn = nn.BatchNorm1d(self.const.weights_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(self.const.weights_dim,self.const.weights_dim)
        self.kappa = nn.Parameter(
            data=torch.FloatTensor([0]),
            requires_grad=True)

    def load_embeddings(self,labels):
        embed_h5py = io.load_h5py_object(self.const.embed_h5py)['embeddings']
        word_to_idx = io.load_json_object(self.const.embed_word_to_idx_json)
        embeddings = np.zeros([len(labels),self.const.embed_dims])
        word_to_label = {}
        for i,label in enumerate(labels):
            if ' ' in label:
                words = label.split(' ')
            elif '_' in label:
                words = label.split('_')
            else:
                words = [label]

            denom = len(words)
            for word in words:
                if word=='tree':
                    denom = len(words)-1
                    continue

                if word not in word_to_label:
                    word_to_label[word] = set()
                word_to_label[word].add(label)

                idx = word_to_idx[word]
                embeddings[i] += embed_h5py[idx][()]
            embeddings[i] /= denom

        if self.const.no_glove == True:
            embeddings[:,:self.const.glove_dim] = 0
        
        self.embed.weight.data.copy_(torch.from_numpy(embeddings))

    def forward(self):
        if self.const.linear==True:
            x = self.fc(self.embed.weight)
        else:
            x = self.fc1(self.embed.weight)
            x = self.bn(x)
            x = self.relu(x)
            x = self.fc2(x)
        return x

    def classify(self,feats,class_weights):
        feats = feats / \
            (1e-6 + torch.pow(torch.sum(feats*feats,1,keepdim=True),0.5))
        class_weights = class_weights / \
            (1e-6 + torch.pow(torch.sum(
                class_weights*class_weights,1,keepdim=True),0.5))
        logits = self.kappa*torch.matmul(feats,class_weights.t())
        return logits

    def margin_loss(self,scores,idx):
        idx_ = idx.data.cpu().numpy()
        mloss = 0
        B = idx.size(0)
        num_samples = 0
        for b in range(B):
            if idx_[b]==-1:
                continue
            
            num_samples += 1
            mloss += torch.sum(
                torch.max(
                    0*scores[b], 
                    0.2 + scores[b] - scores[b,idx_[b]]))

        mloss = mloss / num_samples
        return mloss

    def reverse_loss(self,class_weights):
        y = self.reverse_fc(class_weights)
        d = y - self.embed.weight
        return torch.mean(d*d)


    def sim_loss(self,class_weights):
        class_weights_norm = torch.norm(class_weights,2,1,keepdim=True)
        class_weights = class_weights / (class_weights_norm + 1e-6)

        embed = self.embed.weight
        embed_norm = torch.norm(embed,2,1,keepdim=True)
        embed = embed / (embed_norm + 1e-6)
        
        class_sim = torch.matmul(class_weights,class_weights.t())
        embed_sim = torch.matmul(embed,embed.t())

        delta = class_sim - embed_sim
        return torch.mean(delta*delta)
