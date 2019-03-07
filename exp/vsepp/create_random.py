import numpy as np

glove_npy  = '/home/nfs/tgupta6/Code/vsepp_data/data/coco_precomp/glove_300.npy'
random_npy = '/home/nfs/tgupta6/Code/vsepp_data/data/coco_precomp/random_300.npy'

glove = np.load(glove_npy)
n,d = glove.shape
random = 2*(np.random.rand(n,d)-0.5)
mag = np.sum(np.abs(glove),1)
for i in range(n):
    if mag[i]==0:
        random[i] = 0

import pdb; pdb.set_trace()
np.save(random_npy,random)
