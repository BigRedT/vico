import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import utils.io as io
import utils.pytorch_layers as pytorch_layers
from . embeddings import Embeddings, EmbeddingsConstants


class TransformConstants(io.JsonSerializableClass):
    def __init__(self):
        super(TransformConstants,self).__init__()
        self.in_feat = 100
        self.out_feat = 50


class Transform(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(Transform,self).__init__()
        self.const = copy.deepcopy(const)
        self.fc1 = nn.Linear(self.const.in_feat,self.const.out_feat,bias=False)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(self.const.out_feat,self.const.out_feat)

    def forward(self,x):
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x


class LogBilinearConstants(io.JsonSerializableClass):
    def __init__(self):
        super(LogBilinearConstants,self).__init__()
        self.num_words = 78416
        self.embed_dims = 100
        self.two_embedding_layers = False
        self.cooccur_types = [
            'syn',
            'attr_attr',
            'obj_attr',
            'obj_hyp',
            'context'
        ]
        self.xform_out_feats = [50]*5
        
    @property
    def xform_const(self):
        const = {}
        for i,cooccur_type in enumerate(self.cooccur_types):
            const_ = TransformConstants()
            const_.in_feat = self.embed_dims
            const_.out_feat = self.xform_out_feats[i]
            const[cooccur_type] = const_
        return const

    @property
    def cooccur_type_to_idx(self):
        return {k:i for i,k in enumerate(self.cooccur_types)}


class LogBilinear(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(LogBilinear,self).__init__()
        self.const = copy.deepcopy(const)
        self.embed1 = self.create_embedding_layer()
        if self.const.two_embedding_layers==True:
            self.embed2 = self.create_embedding_layer()
        else:
            self.embed2 = self.embed1
        self.create_transform_layers()

    def create_embedding_layer(self):
        embed_const = EmbeddingsConstants()
        embed_const.num_words = self.const.num_words
        embed_const.embed_dims = self.const.embed_dims
        return Embeddings(embed_const)

    def create_transform_layers(self):
        for cooccur_type in self.const.cooccur_types:
            setattr(
                self,
                'xform_' + cooccur_type,
                Transform(self.const.xform_const[cooccur_type]))

    def forward(self,ids1,ids2,cooccur_type):
        xform = getattr(self,'xform_' + cooccur_type)
        w1,b1 = self.embed1(ids1)
        w2,b2 = self.embed2(ids2)
        w1 = xform(w1)
        w2 = xform(w2)
        i = self.const.cooccur_type_to_idx[cooccur_type]
        scores = torch.sum(w1*w2,1) + b1[:,i] + b2[:,i]
        return scores

    def loss(self,scores,target,x):
        fx = torch.min(0*x+1,x/100)
        gx = x < 100
        gx = gx.float()
        gx_sum = torch.sum(gx) + 1e-6
        gamma = float(np.log(100))+0*scores
        return torch.mean(fx*torch.pow((scores-target),2)) + \
            (torch.sum(gx*torch.max(0*scores,scores-gamma))/gx_sum)



if __name__=='__main__':
    const = LogBilinearConstants()
    const.num_words = 100

    logbilinear = LogBilinear(const).cuda()
    scores = logbilinear([0,1,2],[2,3,4],'syn')
    x = Variable(torch.cuda.FloatTensor([1,2,3]))
    loss = logbilinear.loss(scores,torch.log(x+1e-6))

    import pdb; pdb.set_trace()