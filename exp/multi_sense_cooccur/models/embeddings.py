import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class EmbeddingsConstants(io.JsonSerializableClass):
    def __init__(self):
        self.num_words = 78416
        self.embed_dims = 100
        self.num_cooccur_types = 5


class Embeddings(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(Embeddings,self).__init__()
        self.const = copy.deepcopy(const)
        self.W = nn.Embedding(self.const.num_words,self.const.embed_dims)
        self.b = nn.Embedding(self.const.num_words,self.const.num_cooccur_types)
        self.W.weight.data.mul_(0.01)
        self.b.weight.data.mul_(0.01)

    def forward(self,ids):
        ids = Variable(torch.LongTensor(ids))
        if self.W.weight.is_cuda:
            ids = ids.cuda()
        return self.W(ids), self.b(ids)


if __name__=='__main__':
    const = EmbeddingsConstants()
    const.num_words = 100

    embed = Embeddings(const).cuda()
    import pdb; pdb.set_trace()