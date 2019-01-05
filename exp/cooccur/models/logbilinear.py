import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import utils.io as io
import utils.pytorch_layers as pytorch_layers
from . embeddings import Embeddings, EmbeddingsConstants


class LogBilinearConstants(io.JsonSerializableClass):
    def __init__(self):
        self.num_words = 38000 
        self.embed_dims = 300
        self.two_embedding_layers = True


class LogBilinear(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(LogBilinear,self).__init__()
        self.const = copy.deepcopy(const)
        self.embed1 = self.create_embedding_layer()
        if self.const.two_embedding_layers==True:
            self.embed2 = self.create_embedding_layer()
        else:
            self.embed2 = self.embed1

    def create_embedding_layer(self):
        embed_const = EmbeddingsConstants()
        embed_const.num_words = self.const.num_words
        embed_const.embed_dims = self.const.embed_dims
        return Embeddings(embed_const)

    def forward(self,ids1,ids2):
        w1,b1 = self.embed1(ids1)
        w2,b2 = self.embed2(ids2)
        scores = torch.sum(w1*w2,1) + b1 + b2
        return scores

    def loss(self,scores,target,x,x_max=100,alpha=0.75):
        f_x = 1 #torch.min(0*x+1,torch.pow(x/x_max,alpha))
        return torch.mean(f_x*torch.pow((scores-target),2))
        


class LogBilinearNegPairs(LogBilinear):
    def __init__(self,const):
        super(LogBilinearNegPairs,self).__init__(const)
        
    def forward(self,ids1,ids2):
        w1,b1 = self.embed1(ids1)
        w2,b2 = self.embed2(ids2)
        scores = \
            torch.matmul(w1,torch.transpose(w2,0,1)) + \
            b1 + \
            torch.transpose(b2,0,1)
        return scores

    def loss(self,scores,ids1,ids2,cooccur_mat):
        B = scores.size(0)
        ids1 = Variable(torch.LongTensor(ids1))
        ids2 = Variable(torch.LongTensor(ids2))
        if scores.is_cuda:
            ids1 = ids1.cuda()
            ids2 = ids2.cuda()

        x = cooccur_mat[ids1,:][:,ids2]
        if scores.is_cuda:
            x = x.cuda()
        
        f_x = (x > 0)
        f_x = f_x.float()
        f_x = f_x + 0.01*(1-f_x)
        #f_x = 1 #torch.min(0*x+1,torch.pow(x/x_max,alpha))
        return torch.mean(f_x*torch.pow((scores-torch.log(x+1e-6)),2))

    # def loss(self,scores,words1,words2,cooccur):
    #     B = scores.size(0)
    #     x = np.zeros([B,B])
    #     for i,word1 in enumerate(words1):
    #         context = cooccur[word1]
    #         for j,word2 in enumerate(words2):
    #             if word2 in context:
    #                 x[i,j] = context[word2]
        
    #     x = Variable(torch.FloatTensor(x))
    #     if scores.is_cuda:
    #         x = x.cuda()
        
    #     f_x = 1 #torch.min(0*x+1,torch.pow(x/x_max,alpha))
    #     return torch.mean(f_x*torch.pow((scores-torch.log(x+1e-6)),2))




if __name__=='__main__':
    const = LogBilinearConstants()
    const.num_words = 100

    # logbilinear = LogBilinear(const).cuda()
    # scores = logbilinear([0,1,2],[2,3,4])
    # x = Variable(torch.cuda.FloatTensor([1,2,3]))
    # loss = logbilinear.loss(scores,x)

    logbilinear = LogBilinearNegPairs(const).cuda()
    scores = logbilinear([0,1,2],[2,3,4])
    x = Variable(torch.cuda.FloatTensor([1,2,3]))
    #loss = logbilinear.loss(scores,x)
    import pdb; pdb.set_trace()