# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:17:49 2016

@author: yuga
"""
from chainer import cuda, Variable, FunctionSet, optimizers, serializers
import chainer.functions as F
import cPickle as pickle
from PIL import Image
import numpy as np
from matplotlib import pylab as plt
import cv2
from sklearn import cluster

PICKLE_PATH = "./chainermodel/alex_net.pkl"
IMPUT_IMAGE_PATH = "./2007_000175.jpg"

original_model = pickle.load(open(PICKLE_PATH))

img = np.array(Image.open(IMPUT_IMAGE_PATH))
plt.imshow(img)
img = img.astype(np.float32)
print img.shape
img = img.transpose(2,0,1)
img = img[np.newaxis,:]
img /= 255

print img
print img.shape

model = FunctionSet(conv1=F.Convolution2D(3,  96, 11, stride=4),
                    bn1=F.BatchNormalization(96),
                    conv2=F.Convolution2D(96, 256,  5, pad=2),
                    bn2=F.BatchNormalization(256),
                    conv3=F.Convolution2D(256, 384,  3, pad=1),
                    conv4=F.Convolution2D(384, 384,  3, pad=1),
                    conv5=F.Convolution2D(384, 256,  3, pad=1),
                    fc6=F.Linear(9216, 4096),
                    fc7=F.Linear(4096, 4096),
                    fc8=F.Linear(4096, 1000))

## copy parameter
model.conv1.W.data = original_model.conv1.W.data
model.conv1.b.data = original_model.conv1.b.data
model.conv2.W.data = original_model.conv2.W.data
model.conv2.b.data = original_model.conv2.b.data
model.conv3.W.data = original_model.conv3.W.data
model.conv3.b.data = original_model.conv3.b.data
model.conv4.W.data = original_model.conv4.W.data
model.conv4.b.data = original_model.conv4.b.data
model.conv5.W.data = original_model.conv5.W.data
model.conv5.b.data = original_model.conv5.b.data
model.fc6.W.data = original_model.fc6.W.data
model.fc6.b.data = original_model.fc6.b.data
model.fc7.W.data = original_model.fc7.W.data
model.fc7.b.data = original_model.fc7.b.data

def forward(x_data, train=True, dropout=True):
    
    x = Variable(x_data, volatile=not train)
    a = F.max_pooling_2d(F.relu(model.bn1(model.conv1(x))), 3, stride=2)
    b = F.max_pooling_2d(F.relu(model.bn2(model.conv2(a))), 3, stride=2)
    c = F.relu(model.conv3(b))
    d = F.relu(model.conv4(c))
    e = F.max_pooling_2d(F.relu(model.conv5(d)), 3, stride=2)
    """
    h = F.dropout(F.relu(model.fc6(h)), train=train)
    h = F.dropout(F.relu(model.fc7(h)), train=train)
    """    
    return a,b,c,d,e

conv1,conv2,conv3,conv4,conv5=forward(img)
hypercolumn = []

pos = 1
plt.figure(figsize=(60,60))
for i in xrange(96):
    plt.subplot(12,8,pos)
    img1 = conv1.data[0][i]
    img1 = cv2.resize(img1,(224,224))
    hypercolumn.append(img1)
    plt.imshow(img1)
    pos += 1
plt.show()

pos = 1
plt.figure(figsize=(60,60))
for i in xrange(256):
    plt.subplot(16,16,pos)
    img2 = conv2.data[0][i]
    img2 = cv2.resize(img2,(224,224))
    """
    hypercolumn.append(img2)
    """
    plt.imshow(img2)
    pos += 1
plt.show()

pos = 1
plt.figure(figsize=(60,60))
for i in xrange(384):
    plt.subplot(16,24,pos)
    img3 = conv3.data[0][i]
    img3 = cv2.resize(img3,(224,224))
    """
    hypercolumn.append(img3)
    """
    plt.imshow(img3)
    pos += 1
plt.show()

pos = 1
plt.figure(figsize=(60,60))
for i in xrange(384):
    plt.subplot(16,24,pos)
    img4 = conv4.data[0][i]
    img4 = cv2.resize(img4,(224,224))
    """
    hypercolumn.append(img4)
    """
    plt.imshow(img4)
    pos += 1
plt.show()

pos = 1
plt.figure(figsize=(60,60))
for i in xrange(256):
    plt.subplot(16,16,pos)
    img5 = conv5.data[0][i]
    img5 = cv2.resize(img5,(224,224))
    hypercolumn.append(img5)
    plt.imshow(img5)
    pos += 1
plt.show()

hc = np.array(hypercolumn)
print hc.shape

m = hc.transpose(1,2,0).reshape(50176,-1)

kmeans = cluster.KMeans(n_clusters=35, max_iter=300, n_jobs=5, precompute_distances=True)
cluster_labels = kmeans.fit_predict(m)

imcluster = np.zeros((224,224))
imcluster = imcluster.reshape((224*224,))
imcluster = cluster_labels

plt.imshow(imcluster.reshape(224,224),cmap="hot")