import numpy as np

glove_npy  = '/home/nfs/tgupta6/Code/MAttNet/cache/prepro/refcoco+_unc/glove_300.npy'
random_npy = '/home/nfs/tgupta6/Code/MAttNet/cache/prepro/refcoco+_unc/random_300.npy'

glove = np.load(glove_npy)
n,d = glove.shape
random = 2*(np.random.rand(n,d)-0.5)
mag = np.sum(np.abs(glove),1)
count = 0
for i in range(n):
    if mag[i]==0:
        random[i] = 0
        count +=1

print(count,n)

np.save(random_npy,random)
