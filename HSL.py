import numpy as np
import csv
#np.set_printoptions(threshold=np.inf)

beta = 1.0
dim=1000
K=np.load('./L/WL_%d.npy'%(dim))

P=np.sum(K,axis=0)
P=np.diag(P)
print(P)

L=P-K
print(L)
#with open('a.csv','ab') as f:
#    writer=csv.writer(f)
#    writer.writerows(K)

#np.save('./L_%.2f/WL_%d.npy'%(beta,dim),L)
#print "save file to ./L_%.2f/WL_%d.npy" % (beta, dim)
np.save('./L_B/WL_%d.npy'%(dim),L)
