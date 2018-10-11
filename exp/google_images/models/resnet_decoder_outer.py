import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class ResnetDecoderOuterConstants(io.JsonSerializableClass):
    def __init__(self):
        self.input_dims = 960
        self.output_size = [224,224]


class ResnetDecoderOuter(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(ResnetDecoderOuter,self).__init__()
        self.const = copy.deepcopy(const)
        self.layers = nn.Sequential(
            nn.Conv2d(self.const.input_dims,512,3),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512,256,3),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Upsample(size=[32,32],mode='bilinear'),
            nn.Conv2d(256,128,3),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Upsample(size=[64,64],mode='bilinear'),
            nn.Conv2d(128,64,3),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Upsample(size=[128,128],mode='bilinear'),
            nn.Conv2d(64,32,3),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Upsample(size=self.const.output_size,mode='bilinear'),
            nn.Conv2d(32,3,3,1,1),
            nn.Tanh())
        self.l1_criterion = nn.L1Loss()
        self.l2_criterion = nn.MSELoss()

    def forward(self,x):
        x = self.layers(x)
        return x

    def compute_loss(self,x,y,criterion='l1'):
        y = Variable(y.data)
        if criterion=='l1':
            loss = self.l1_criterion(x,y)
        elif criterion=='l2':
            loss = self.l2_criterion(x,y)
        else:
            assert_str = 'Criterion not implemented'
            assert(False), assert_str
        return loss

