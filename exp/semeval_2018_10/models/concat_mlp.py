import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class ConcatMLPConstants(io.JsonSerializableClass):
    def __init__(self):
        self.embedding_dim = 300
        self.layer_units = []
        
    @property
    def mlp_const(self):
        factor_const = {
            'in_dim': 3*self.embedding_dim,
            'out_dim': 1,
            'out_activation': 'Sigmoid',
            'layer_units': self.layer_units,
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True,
            'drop_prob': 0,
        }
        return factor_const


class ConcatMLP(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(ConcatMLP,self).__init__()
        self.const = copy.deepcopy(const)
        self.mlp = pytorch_layers.create_mlp(self.const.mlp_const)

    def forward(
            self,
            word1_embedding,
            word2_embedding,
            feature_embedding):
        x = torch.cat((
            word1_embedding,
            word2_embedding,
            feature_embedding),1)
        return self.mlp(x)
