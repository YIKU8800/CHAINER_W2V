import numpy as np
import csv
#np.set_printoptions(threshold=np.inf)

rate=10
dim=40
K=np.load('K_%d.npy'%(dim))
K=np.exp(-K*rate)
print K

P=np.sum(K,axis=0)
P=np.diag(P)
print P

L=P-K
print L
#with open('a.csv','ab') as f:
#    writer=csv.writer(f)
#    writer.writerows(K)

np.save('./L_%d/L_%d.npy'%(rate,dim),L)
