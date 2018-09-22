import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class ConcatSVMConstants(io.JsonSerializableClass):
    def __init__(self):
        self.embedding_dim = 300
        self.layer_units = []
        self.drop_prob = 0
        self.out_drop_prob = 0
        self.use_bn = True
        self.l2_weight = 0 
        self.use_embedding_linear_feats = True
        self.use_embedding_quadratic_feats = True
        self.use_distance_linear_feats = True
        self.use_distance_quadratic_feats = True
        
    @property
    def mlp_const(self):
        factor_const = {
            'in_dim': 6*7 + 0*6*self.embedding_dim,
            'out_dim': 1,
            'out_activation': 'Identity',
            'layer_units': self.layer_units,
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': self.use_bn,
            'drop_prob': self.drop_prob,
            'out_drop_prob': self.out_drop_prob,
        }
        return factor_const


class ConcatSVM(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(ConcatSVM,self).__init__()
        self.const = copy.deepcopy(const)
        self.mlp = pytorch_layers.create_mlp(self.const.mlp_const)
        self.bn = nn.BatchNorm1d(6)

    def forward(
            self,
            word1_embedding,
            word2_embedding,
            feature_embedding):
        B = word1_embedding.size(0)
        cs_word1_feat = self.cosine_similarity(word1_embedding,feature_embedding)
        cs_word2_feat = self.cosine_similarity(word2_embedding,feature_embedding)
        cs_word1_word2 = self.cosine_similarity(word1_embedding,word2_embedding)
        l1_word1_feat = self.l1_norm(word1_embedding-feature_embedding)
        l1_word2_feat = self.l1_norm(word2_embedding-feature_embedding)
        l1_word1_word2 = self.l1_norm(word1_embedding-word2_embedding)
        distance_feats = 1*torch.cat((
            cs_word1_feat,
            cs_word2_feat,
            cs_word1_word2,
            l1_word1_feat/(1*self.const.embedding_dim),
            l1_word2_feat/(1*self.const.embedding_dim),
            l1_word1_word2/(1*self.const.embedding_dim)),1)
        #distance_feats = self.bn(distance_feats)
        distance_quadratic_feats = \
            torch.unsqueeze(distance_feats,1) * \
            torch.unsqueeze(distance_feats,2)
        distance_quadratic_feats = distance_quadratic_feats.view(B,-1)
        embedding_feats = torch.cat((
            word1_embedding,
            word2_embedding,
            feature_embedding),1)
        embedding_quadratic_feats = torch.cat((
            word1_embedding*feature_embedding,
            word2_embedding*feature_embedding,
            word1_embedding*word2_embedding),1)
        x = torch.cat((
            # self.const.use_embedding_linear_feats*embedding_feats,
            # self.const.use_embedding_quadratic_feats*embedding_quadratic_feats,
            self.const.use_distance_linear_feats*distance_feats,
            self.const.use_distance_quadratic_feats*distance_quadratic_feats),1)
        score = self.mlp(x)[:,0] # Convert Bx1 to B
        return score

    def compute_hinge_loss(self,score,label):
        scaled_label = 2*label-1
        return torch.mean(torch.max(0*score,1-scaled_label*score))

    def compute_l2_loss(self):
        l2_reg = None
        for name, W in self.named_parameters():
            if 'bias' in name:
                continue
            if 'embedding' in name:
                continue
            if l2_reg is None:
                l2_reg = W.pow(2).sum()
            else:
                l2_reg = l2_reg + W.pow(2).sum()
        return l2_reg

    def compute_loss(self,score,label):
        hinge_loss = self.compute_hinge_loss(score,label)
        l2_reg = self.compute_l2_loss()
        total_loss = hinge_loss + (self.const.l2_weight * l2_reg)
        return total_loss, hinge_loss, l2_reg

    def cosine_similarity(self,a,b):
        a_norm = torch.norm(a,2,1,keepdim=True)
        b_norm = torch.norm(b,2,1,keepdim=True)
        return torch.sum(a*b,1,keepdim=True)/(a_norm*b_norm+1e-6)

    def l1_norm(self,x):
        return torch.sum(torch.abs(x),1,keepdim=True)