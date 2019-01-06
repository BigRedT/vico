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
        self.embed_dims = 300
        self.embed_h5py = None
        self.embed_word_to_idx_json = None
        self.weights_dim = 64
        self.linear = True


class Embed2Class(nn.Module):
    def __init__(self,const):
        super(Embed2Class,self).__init__()
        self.const = copy.deepcopy(const)
        self.embed = nn.Embedding(
            self.const.num_classes,
            self.const.embed_dims)
        if self.const.linear==True:
            self.fc = nn.Linear(self.const.embed_dims,self.const.weights_dim)
        else:
            self.fc1 = nn.Linear(self.const.embed_dims,self.const.weights_dim)
            self.bn = nn.BatchNorm1d(self.const.weights_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(self.const.weights_dim,self.const.weights_dim)

    def load_embeddings(self,labels):
        embed_h5py = io.load_h5py_object(self.const.embed_h5py)['embeddings']
        word_to_idx = io.load_json_object(self.const.embed_word_to_idx_json)
        embeddings = np.zeros([len(labels),self.const.embed_dims])
        for i,label in enumerate(labels):
            if ' ' in label:
                words = label.split(' ')
            elif '_' in label:
                words = label.split('_')
            else:
                words = [label]
            for word in words:
                idx = word_to_idx[word]
                embeddings[i] += embed_h5py[idx][()]
            embeddings[i] /= len(words)
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
        logits = torch.matmul(feats,class_weights.t())
        return logits

