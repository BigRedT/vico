import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class WordClassifierLayerConstants(io.JsonSerializableClass):
    def __init__(self):
        self.input_dim = 2048
        self.num_classes = 8078
        self.layer_units = []
        
    @property
    def mlp_const(self):
        factor_const = {
            'in_dim': self.input_dim,
            'out_dim': self.num_classes,
            'out_activation': 'Identity',
            'layer_units': self.layer_units,
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True,
            'drop_prob': 0,
            'out_drop_prob': 0,
        }
        return factor_const


class WordClassifierLayer(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(WordClassifierLayer,self).__init__()
        self.const = const
        self.mlp = pytorch_layers.create_mlp(self.const.mlp_const)

    def forward(self,x):
        return self.mlp(x)