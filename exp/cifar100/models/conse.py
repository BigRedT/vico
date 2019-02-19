import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import utils.io as io
import utils.pytorch_layers as pytorch_layers

class Conse():
    def __init__(self,k,embed,held_out_idx):
        self.k = k
        self.held_out_idx = held_out_idx
        self.embed = embed
        self.embed_n = embed / \
            torch.pow(torch.sum(embed*embed,1,keepdim=True),0.5)
        sim = torch.matmul(self.embed_n,self.embed_n.t()) # 100x100
        for i in self.held_out_idx:
            sim[:,i] = sim[:,i]*0 - 1
        top_sim, top_idxs = torch.topk(sim,self.k,1)
        alpha = top_sim / torch.sum(top_sim,1,keepdim=True)
        #alpha = 0.5*(top_sim + 1) / self.k
        self.alpha = alpha.data.cpu().numpy()
        self.top_idxs = top_idxs.data.cpu().numpy()

    def infer_prob(self,prob):
        prob_new_ = prob.data.cpu().numpy()
        prob_new = np.copy(prob_new_)
        for i in self.held_out_idx:
            prob_new[:,i] = np.sum(
                prob_new_[:,self.top_idxs[i]]*self.alpha[i],
                1,
                keepdims=False)

        return prob_new

    def infer_prob2(self,prob):
        prob_ = prob.clone()
        for i in self.held_out_idx:
            prob_[:,i] = 0
        top_prob,top_idxs = torch.topk(prob_,self.k,1)
        top_prob = top_prob / torch.sum(top_prob,1,keepdim=True)
        embed_new = []
        for i in range(prob.size(0)):
            embed_new.append(torch.matmul(
                top_prob[i].view(1,-1),
                self.embed_n[top_idxs[i],:]))

        embed_new = torch.cat(embed_new,0)
        embed_new = embed_new / \
            torch.pow(torch.sum(embed_new*embed_new,1,keepdim=True),0.5)
        prob_new = 0.5*(1+torch.matmul(embed_new,self.embed_n.t()))
        for i in range(prob.size(1)):
            if i not in self.held_out_idx:
                prob_new[:,i] = 0*prob[:,i]

        return prob_new.data.cpu().numpy()
                
