from chainer import Variable, FunctionSet, optimizers,serializers,cuda
import chainer.functions  as F
import numpy as np
#np.set_printoptions(threshold=np.inf)
import time
import cPickle as pickle
import csv

np.random.seed(111111)
save_name='freq_sheet.csv'

train=np.load('y_train.npy')
train=train.astype(np.int32)

print train.shape

matrix=np.zeros([260,260])
for i in xrange(len(train)):
    index= np.where(train[i]==1.0)[0]
    for j in xrange(len(index)):
        matrix[index[j]][index]+=1
print matrix

#np.save('./freq_sheet.npy',matrix)
with open(save_name,'ab') as f:
    writer=csv.writer(f)
    writer.writerows(matrix)

