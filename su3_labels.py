# -*- coding: utf-8 -*-
"""
Created on Fri May  2 08:27:25 2014

@author: kurita
"""

import numpy as np
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import MySU3
import csv

word='bear'
dim=40

data=np.load('y_train.npy')
#data=np.load('freq_sheet.npy')

E, U, V = MySU3.su3(data)

print 'data\n', data

E=np.array(E)
U=np.array(U)
V=np.array(V)

print 'E',E.shape,'\n', E

print 'U',U.shape,'\n', U

print 'V',V.shape,'\n', V
"""
print '====== transposed data ======='

E, U, V = MySU3.su3(data.T)

print 'data\n', data

print 'E',E.shape,'\n', E

print 'U',U.shape,'\n', U

print 'V',V.shape,'\n', V
"""
V=V[:,0:dim]
print V.shape
"""
for i in xrange(260):
    summ=0
    summ=np.sum(V[i,:])
    V[i]=V[i]/summ
"""
"""
elements=0
total=np.sum(E)
scores=[]
for i in xrange(259):
    elements+=E[i]
    scores.append(elements/total)
scores=np.array(scores)
for i in xrange(259):
    print i,scores[i]
"""
label=[]
index={}
f=open("corel5k_words.txt")
line=f.read()
f.close()
line=line.split('\n')
del line[260]
for i in xrange(len(line)):
    label.append(line[i])
    index[line[i]]=i

distance=np.zeros(260)
for i in xrange(260):
    distance[i]=np.sum((V[index[word]]-V[i])**2)
    distance=np.array(distance)

count=0
for i in distance.argsort():
    if count==0:
        print label[i],i
    else:
        print label[i],distance[i],i
    if count>5:
        break
    count+=1



distance=np.ones([260,260])
for i in xrange(260):
    for j in xrange(260):
        squared_error=np.sum((V[i]-V[j])**2)
	distance[i][j]=squared_error
print distance

np.save('./../L/|q|_%d.npy'%dim,distance)
