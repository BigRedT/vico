import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class ResnetDecoderInnerConstants(io.JsonSerializableClass):
    def __init__(self):
        self.input_dims = 128*4*4
        self.middle_dims = 128
        self.output_dims = 960


class ResnetDecoderInner(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(ResnetDecoderInner,self).__init__()
        self.const = copy.deepcopy(const)
        self.fc = nn.Linear(
            self.const.input_dims,
            self.const.middle_dims*4*4) # 128*4*4
        self.layers = nn.Sequential(
            nn.BatchNorm2d(self.const.middle_dims),
            nn.ELU(),
            nn.Upsample(size=[8,8],mode='bilinear'),
            nn.Conv2d(self.const.middle_dims,self.const.middle_dims//2,3,1,1), # 64x8x8
            nn.BatchNorm2d(self.const.middle_dims//2),
            nn.ELU(),
            nn.Upsample(size=[16,16],mode='bilinear'),
            nn.Conv2d(self.const.middle_dims//2,self.const.output_dims//32,3,1,1), # 30x16x16
            nn.BatchNorm2d(self.const.output_dims//32),
            nn.ELU(),
            nn.Conv2d(self.const.output_dims//32,self.const.output_dims//16,1,1,0), # 60x16x16
            nn.BatchNorm2d(self.const.output_dims//16),
            nn.ELU(),
            nn.Conv2d(self.const.output_dims//16,self.const.output_dims//8,1,1,0), # 120x16x16
            nn.BatchNorm2d(self.const.output_dims//8),
            nn.ELU(),
            nn.Conv2d(self.const.output_dims//8,self.const.output_dims//4,1,1,0), # 240x16x16
            nn.BatchNorm2d(self.const.output_dims//4),
            nn.ELU(),
            nn.Conv2d(self.const.output_dims//4,self.const.output_dims//2,1,1,0), # 480x16x16
            nn.BatchNorm2d(self.const.output_dims//2),
            nn.ELU(),
            nn.Conv2d(self.const.output_dims//2,self.const.output_dims,1,1,0))   # 960x16x16
        self.l1_criterion = nn.L1Loss()
        self.l2_criterion = nn.MSELoss()

    def forward(self,x):
        x = self.fc(x)
        B = x.size(0)
        x = self.layers(x.view(B,-1,4,4))
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

