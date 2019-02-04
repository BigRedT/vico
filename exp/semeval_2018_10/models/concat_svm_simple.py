import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class ConcatSVMConstants(io.JsonSerializableClass):
    def __init__(self):
        self.embedding_dim = 300 # whole embedding dimension
        self.glove_dim = 300    # glove part of the whole embeddings
        self.layer_units = []
        self.drop_prob = 0
        self.out_drop_prob = 0
        self.use_bn = True
        self.use_out_bn = False
        self.l2_weight = 0 
        self.use_distance_linear_feats = True
        self.use_distance_quadratic_feats = True
        self.visual_only = False
        
    @property
    def mlp_const(self):
        if self.embedding_dim==self.glove_dim:
            in_dim = 6*7
        else:
            in_dim = 2*6*7
        factor_const = {
            'in_dim': in_dim,
            'out_dim': 1,
            'out_activation': 'Identity',
            'layer_units': self.layer_units,
            'activation': 'ReLU',
            'use_bn': self.use_bn,
            'use_out_bn': self.use_out_bn,
            'drop_prob': self.drop_prob,
            'out_drop_prob': self.out_drop_prob,
        }
        return factor_const


class ConcatSVM(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(ConcatSVM,self).__init__()
        self.const = copy.deepcopy(const)
        self.mlp = pytorch_layers.create_mlp(self.const.mlp_const)
        self.bn = nn.BatchNorm1d(self.const.mlp_const['in_dim'])
        print('2D-SVM')
        self.w = nn.Parameter(torch.FloatTensor([0,0,0]))

    def forward(
            self,
            word1_embedding,
            word2_embedding,
            feature_embedding):
        B = word1_embedding.size(0)
        score_glove = self.compute_distance_features_glove(
            word1_embedding[:,:self.const.glove_dim],
            word2_embedding[:,:self.const.glove_dim],
            feature_embedding[:,:self.const.glove_dim])
        if self.const.embedding_dim > self.const.glove_dim:
            score_visual = self.compute_distance_features_visual(
                word1_embedding[:,self.const.glove_dim:],
                word2_embedding[:,self.const.glove_dim:],
                feature_embedding[:,self.const.glove_dim:])
            if self.const.visual_only==True:
                score = self.w[1]*score_visual + self.w[2]
            else:
                score = \
                    self.w[0]*score_glove + \
                    self.w[1]*score_visual + \
                    self.w[2]       
        else:
            score = self.w[0]*score_glove + self.w[2]

        return score

    def compute_distance_features_glove(self,word1,word2,feature):
        B = word1.size(0)
        cs_word1_feat = self.cosine_similarity(word1,feature)
        cs_word2_feat = self.cosine_similarity(word2,feature)
        cs_word1_word2 = self.cosine_similarity(word1,word2)
        x = cs_word1_feat - cs_word2_feat
        return x

    def compute_distance_features_visual(self,word1,word2,feature):
        B = word1.size(0)
        cs_word1_feat = self.cosine_similarity(word1,feature)
        cs_word2_feat = self.cosine_similarity(word2,feature)
        cs_word1_word2 = self.cosine_similarity(word1,word2)
        x = cs_word1_feat - cs_word2_feat
        return x


    def compute_hinge_loss(self,score,label):
        scaled_label = 2*label-1
        return torch.mean(torch.max(0*score,0.2-scaled_label*score[:,0]))

    def compute_l2_loss(self):
        l2_reg = self.w[:2].pow(2).sum()
        return l2_reg

    def compute_loss(self,score,label):
        hinge_loss = self.compute_hinge_loss(score,label)
        l2_reg = self.const.l2_weight*self.compute_l2_loss()
        total_loss = hinge_loss + l2_reg
        return total_loss, hinge_loss, l2_reg

    def cosine_similarity(self,a,b):
        a_norm = torch.norm(a,2,1,keepdim=True)
        b_norm = torch.norm(b,2,1,keepdim=True)
        return torch.sum(a*b,1,keepdim=True)/(a_norm*b_norm+1e-6)

    def l1_norm(self,x):
        return torch.sum(torch.abs(x),1,keepdim=True)