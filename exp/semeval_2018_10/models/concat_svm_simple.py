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

    def forward(
            self,
            word1_embedding,
            word2_embedding,
            feature_embedding):
        B = word1_embedding.size(0)
        distance_feats_glove = self.compute_distance_features(
            word1_embedding[:,:self.const.glove_dim],
            word2_embedding[:,:self.const.glove_dim],
            feature_embedding[:,:self.const.glove_dim])
        if self.const.embedding_dim > self.const.glove_dim:
            distance_feats_visual = self.compute_distance_features(
                word1_embedding[:,self.const.glove_dim:],
                word2_embedding[:,self.const.glove_dim:],
                feature_embedding[:,self.const.glove_dim:])
            distance_feats = torch.cat((
                distance_feats_glove,
                distance_feats_visual),1)
        else:
            distance_feats = distance_feats_glove
            
        x = self.bn(distance_feats)
        score = self.mlp(x)[:,0] # Convert Bx1 to B
        return score

    def compute_distance_features(self,word1,word2,feature):
        B = word1.size(0)
        cs_word1_feat = self.cosine_similarity(word1,feature)
        cs_word2_feat = self.cosine_similarity(word2,feature)
        cs_word1_word2 = self.cosine_similarity(word1,word2)
        l1_word1_feat = self.l1_norm(word1-feature)
        l1_word2_feat = self.l1_norm(word2-feature)
        l1_word1_word2 = self.l1_norm(word1-word2)
        distance_feats = torch.cat((
            cs_word1_feat,
            cs_word2_feat,
            cs_word1_word2,
            l1_word1_feat/(self.const.embedding_dim),
            l1_word2_feat/(self.const.embedding_dim),
            l1_word1_word2/(self.const.embedding_dim)),1)
        distance_quadratic_feats = \
            torch.unsqueeze(distance_feats,1) * \
            torch.unsqueeze(distance_feats,2)
        distance_quadratic_feats = distance_quadratic_feats.view(B,-1)
        x = torch.cat((
            self.const.use_distance_linear_feats*distance_feats,
            self.const.use_distance_quadratic_feats*distance_quadratic_feats),1)
        return x

    def compute_hinge_loss(self,score,label):
        scaled_label = 2*label-1
        return torch.mean(torch.max(0*score,1-scaled_label*score))

    def compute_l2_loss(self):
        # l2_reg = None
        # for name, W in self.named_parameters():
        #     if 'bias' in name:
        #         continue
        #     if 'embedding' in name:
        #         continue
        #     if l2_reg is None:
        #         l2_reg = W.pow(2).sum()
        #     else:
        #         l2_reg = l2_reg + W.pow(2).sum()
        #     import pdb; pdb.set_trace()
        # return l2_reg
        W = self.mlp.layers[0][0].weight
        l2_reg = W.pow(2).sum()
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