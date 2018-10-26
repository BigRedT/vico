import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class DecoderConstants(io.JsonSerializableClass):
    def __init__(self):
        self.input_dims = 300
        self.output_dims = 2048


class Decoder(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(Decoder,self).__init__()
        self.const = copy.deepcopy(const)
        self.layers = nn.Sequential(
            nn.Linear(self.const.input_dims,self.const.output_dims))
        self.l1_criterion = nn.L1Loss()
        self.l2_criterion = nn.MSELoss()

    def forward(self,x):
        x = self.layers(x)
        return x

    def compute_loss(self,x,y,criterion='l2'):
        y = Variable(y.data)
        if criterion=='l1':
            loss = self.l1_criterion(x,y)
        elif criterion=='l2':
            loss = self.l2_criterion(x,y)
        else:
            assert_str = 'Criterion not implemented'
            assert(False), assert_str
        return loss