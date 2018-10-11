import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class ResnetEncoderInnerConstants(io.JsonSerializableClass):
    def __init__(self):
        self.input_dims = 512 + 256 + 128 + 64 # 960
        self.middle_dims = 128
        self.output_dims = 128*4*4 # 2048


class ResnetEncoderInner(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(ResnetEncoderInner,self).__init__()
        self.const = copy.deepcopy(const)
        self.layers = nn.Sequential(
            nn.Conv2d(self.const.input_dims,self.const.input_dims//2,1,1,0),   # 480x16x16
            nn.BatchNorm2d(self.const.input_dims//2),
            nn.ELU(),
            nn.Conv2d(self.const.input_dims//2,self.const.input_dims//4,1,1,0),   # 240x16x16
            nn.BatchNorm2d(self.const.input_dims//4),
            nn.ELU(),
            nn.Conv2d(self.const.input_dims//4,self.const.input_dims//8,1,1,0),   # 120x16x16
            nn.BatchNorm2d(self.const.input_dims//8),
            nn.ELU(),
            nn.Conv2d(self.const.input_dims//8,self.const.input_dims//16,1,1,0),   # 60x16x16
            nn.BatchNorm2d(self.const.input_dims//16),
            nn.ELU(),
            nn.Conv2d(self.const.input_dims//16,self.const.input_dims//32,1,1,0),   # 30x16x16
            nn.BatchNorm2d(self.const.input_dims//32),
            nn.ELU(),
            nn.Conv2d(self.const.input_dims//32,self.const.middle_dims//2,3,2,1),   # 64x8x8
            nn.BatchNorm2d(self.const.middle_dims//2),
            nn.ELU(),
            nn.Conv2d(self.const.middle_dims//2,self.const.middle_dims,3,2,1), # 128x4x4
            nn.BatchNorm2d(self.const.middle_dims),
            nn.ELU())
        self.fc = nn.Linear(
            4*4*self.const.middle_dims,
            self.const.output_dims) # 128*4*4

    def forward(self,x):
        x = self.layers(x)
        B,C,H,W = x.size()
        x = self.fc(x.view(B,-1))
        x_norm = \
            x / \
            (1e-6 + torch.norm(x,p=2,dim=1,keepdim=True))
        return x, x_norm

