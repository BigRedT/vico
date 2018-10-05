import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class DecoderConstants(io.JsonSerializableClass):
    def __init__(self):
        self.input_dim = 2048
        self.output_size = [224,224]


class Decoder(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(Decoder,self).__init__()
        self.const = copy.deepcopy(const)
        self.fc = nn.Linear(self.const.input_dim,256*8*8)
        self.conv_upsample_layers = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256,128,3),
            nn.Upsample(size=[16,16],mode='bilinear'),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128,64,3),
            nn.Upsample(size=[32,32],mode='bilinear'),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64,32,3),
            nn.Upsample(size=[64,64],mode='bilinear'),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32,16,3),
            nn.Upsample(size=[128,128],mode='bilinear'),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16,8,3),
            nn.Upsample(size=self.const.output_size,mode='bilinear'),
            nn.ELU(),
            nn.Conv2d(8,3,3,padding=1),
            nn.Tanh())
        self.l1_loss = nn.L1Loss()

    def forward(self,x):
        x = self.fc(x)
        x = x.view(-1,256,8,8)
        x = self.conv_upsample_layers(x)
        return x

    def recon_loss(self,x,y):
        return self.l1_loss(x,y)

    def high_freq_recon_loss(self,x,y,k=100):
        B = x.size(0)
        value,_ = torch.topk(torch.abs(x-y).view(B,-1),k=k,dim=-1)
        return torch.mean(value)
