import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import utils.io as io
import utils.pytorch_layers as pytorch_layers
from .embeddings import Embeddings, EmbeddingsConstants


class TransformConstants(io.JsonSerializableClass):
    def __init__(self):
        super(TransformConstants,self).__init__()
        self.in_feat = 100
        self.out_feat = 50
        self.identity = False
        self.xform_type = 'linear' # identity | linear | affine | nonlinear | select
        # begin and end only used with select
        self.begin = 0
        self.end = 99
        # num_layers only used with nonlinear
        self.num_layers = None


class Transform(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(Transform,self).__init__()
        self.const = copy.deepcopy(const)
        self.layers = self.construct_layers()

    def construct_layers(self):
        if self.const.xform_type=='linear':
            return nn.Sequential(
                nn.Linear(self.const.in_feat,self.const.out_feat,bias=False))
        elif self.const.xform_type=='affine':
            return nn.Sequential(
                nn.Linear(self.const.in_feat,self.const.out_feat))
        elif self.const.xform_type=='select':
            return None
        elif self.const.xform_type=='identity':
            return None
        elif self.const.xform_type=='nonlinear':
            if self.const.num_layers==2:
                out_feat_ = (self.const.in_feat + self.const.out_feat)//2
                return nn.Sequential(
                    nn.Linear(self.const.in_feat,out_feat_),
                    nn.Tanh(), #nn.ReLU(),
                    nn.Linear(out_feat_,self.const.out_feat))
            elif self.const.num_layers==4:
                d = (self.const.out_feat - self.const.in_feat)//4
                in_feat = self.const.in_feat
                return nn.Sequential(
                    nn.Linear(in_feat,in_feat+d),
                    nn.Tanh(), #nn.ReLU(),
                    nn.Linear(in_feat+d,in_feat+2*d),
                    nn.Tanh(), #nn.ReLU(),
                    nn.Linear(in_feat+2*d,in_feat+3*d),
                    nn.Tanh(), #nn.ReLU(),
                    nn.Linear(in_feat+3*d,self.const.out_feat))
            else:
                err_msg = f'Not implemented num_layers {self.const.num_layers}'
                assert(False), err_msg
        else:
            err_msg = f'Not implemented num_layers {self.const.xform_type}'
            assert(False), err_msg
                

    def forward(self,x):
        if self.const.xform_type=='select':
            return x[:,self.const.begin:self.const.end+1]

        if self.const.xform_type=='identity':
            return x
        
        return self.layers(x)


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
        self.xform_out_feat_dim = 50
        self.xform_type = 'linear'
        # xform_num_layers used on if type is 'nonlinear
        self.xform_num_layers = 2
        self.use_bias = False
        self.use_fx = False

    @property
    def xform_out_feats(self):
        return [self.xform_out_feat_dim]*len(self.cooccur_types)

    @property
    def xform_const(self):
        const = {}
        for i,cooccur_type in enumerate(self.cooccur_types):
            const_ = TransformConstants()
            
            if self.xform_type=='select':
                const_.in_feat = self.embed_dims // len(self.cooccur_types)
                const_.begin = i*const_.in_feat
                const_.end = (i+1)*const_.in_feat - 1
            else:
                const_.in_feat = self.embed_dims
                const_.begin = None
                const_.end = None
            
            if self.xform_type=='identity':
                const_.out_feat = const_.in_feat
            else:
                const_.out_feat = self.xform_out_feats[i]
                
            const_.xform_type = self.xform_type
            const_.num_layers = self.xform_num_layers

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
        scores = torch.sum(w1*w2,1)
        if self.const.use_bias==True:
            scores = scores + b1[:,i] + b2[:,i]
        return scores

    def loss(self,scores,target,x):
        if self.const.use_fx==False:
            fx = 1
        else:
            fx = torch.min(0*x+1,x/50.0)
        return torch.mean(fx*torch.pow((scores-target),2))



if __name__=='__main__':
    const = LogBilinearConstants()
    const.num_words = 100

    logbilinear = LogBilinear(const).cuda()
    scores = logbilinear([0,1,2],[2,3,4],'syn')
    x = Variable(torch.cuda.FloatTensor([1,2,3]))
    loss = logbilinear.loss(scores,torch.log(x+1e-6))

    import pdb; pdb.set_trace()