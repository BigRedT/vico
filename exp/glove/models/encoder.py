import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class EncoderConstants(io.JsonSerializableClass):
    def __init__(self):
        self.input_dims = 2048
        self.output_dims = 300


class Encoder(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(Encoder,self).__init__()
        self.const = copy.deepcopy(const)
        self.layers = nn.Sequential(
            nn.Linear(self.const.input_dims,self.const.output_dims))

    def forward(self,x):
        x = self.layers(x)
        x_norm = \
            x / \
            (1e-6 + torch.norm(x,p=2,dim=1,keepdim=True))
        return x, x_norm