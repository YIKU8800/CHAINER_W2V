
# coding: utf-8

# In[240]:

from PIL import Image
import numpy as np
import chainer
from chainer import Variable,serializers, cuda
import chainer.functions  as F
import chainer.links as CL
import cupy
from collections import Counter
np.random.seed(0)
import sklearn.metrics


# In[241]:

def load_image(path, width=0, height=0):
    img = Image.open(path).convert('RGB')
    data = np.array(img)
    if width == 0:
        width = data.shape[0]
        height = data.shape[1]
    return data.reshape(width, height, 3)


# In[242]:

def gen_test_co():
    test_co = {}
    test_idx = {}
    idx = 0
    with open('corel5k/corel5k_test_list.txt') as f:
        for line in f:
            
            test_idx[line.strip('\n')] = idx
            idx = idx + 1
            
            t = line.strip('\n').split('/')
            if not t[0] in test_co:
                test_co[t[0]] = []
            test_co[t[0]].append(t[1])
            
    return test_co, test_idx


def load_labels():
    labelidx = {}
    f = open("corel5k_words.txt")
    line = f.read()
    f.close()
    line = line.split('\n')
    del line[260]
    for i in range(len(line)):
        labelidx[line[i]] = i

    return labelidx


labelidx = load_labels()

def load_gt_labels():
    gt_labels = []
    with open("ground_truth_labels.txt") as f:
        for line in f:
            t = line.strip('\n')

            labels = np.zeros(260)
            for label in (t.split('|')):
                idx = labelidx[label]
                labels[idx] = 1

            gt_labels.append(labels)
    return gt_labels


gt_labels = load_gt_labels()


# In[243]:

def get_test_img_idx(test_idx, idir, iid):
    t = '%s/%s' % (idir, iid)
    idx = test_idx[t]
    return idx

test_co, test_idx = gen_test_co()


# In[244]:

def load_labels():
    label=[]
    f=open("corel5k_words.txt")
    line=f.read()
    f.close()
    line=line.split('\n')
    del line[260]
    for i in range(len(line)):
        label.append(line[i])
    return label

labels = load_labels()

def load_image(filepath):
    filepath = "corel5k/%s" % filepath
    x_train = [np.array(Image.open(filepath).resize((im_size,im_size)))]
    x_train=np.array(x_train)
    x_train=x_train.astype(np.int32)/255.0
    x_train=np.transpose(x_train,(0,3,1,2))
    x_train=x_train.astype(np.float32)
    return x_train

y_test = np.load('corel5k/y_test.npy')
y_test = y_test.astype(np.int32)


# In[245]:

def get_origin_labels(idir, iid):
    idx = get_test_img_idx(test_idx, idir, iid)
    return y_test[idx]

# idx from 0
def get_gt_labels(idx):
    return gt_labels[idx]

def get_orgin_label_str(idir, iid):
    idx = get_test_img_idx(test_idx, idir, iid)
    #path = '%s/%s' % (idir, iid)
    r = ''
    for i in range(0,260):
        if y_test[idx][i] == 1:
            if r == '':
                r += '%s' % labels[i]
            else:
                r += '|%s' % labels[i]
    
    return r


# In[246]:

im_size=127

threshold = 0.1

model_path = {
        #'without' : 'pre_model/paper/without.model',
        'without' : 'model/w2v_model_0.001000_1000.model',
        #'word2vec' : 'model/ohw2v_model_0.0000050000_1.00_1000.model',}
        'word2vec' : 'model/w2v_model_0.001000_1000.model',}


without_model = chainer.Chain(conv1=CL.Convolution2D(3,  96, 11, stride=4),
                    bn1=CL.BatchNormalization(96),
                    conv2=CL.Convolution2D(96, 256,  5, pad=2),
                    bn2=CL.BatchNormalization(256),
                    conv3=CL.Convolution2D(256, 384,  3, pad=1),
                    conv4=CL.Convolution2D(384, 384,  3, pad=1),
                    conv5=CL.Convolution2D(384, 256,  3, pad=1),
                    fc6=CL.Linear(2304,1024),
                    fc7=CL.Linear(1024, 260))

serializers.load_npz(model_path['without'], without_model)

word2vec_model = chainer.Chain(conv1=CL.Convolution2D(3,  96, 11, stride=4),
                    bn1=CL.BatchNormalization(96),
                    conv2=CL.Convolution2D(96, 256,  5, pad=2),
                    bn2=CL.BatchNormalization(256),
                    conv3=CL.Convolution2D(256, 384,  3, pad=1),
                    conv4=CL.Convolution2D(384, 384,  3, pad=1),
                    conv5=CL.Convolution2D(384, 256,  3, pad=1),
                    fc6=CL.Linear(2304,1024),
                    fc7=CL.Linear(1024, 260))

serializers.load_npz(model_path['word2vec'], word2vec_model)


def predict(model, x_data):
    #x = Variable(cuda.to_gpu(x_data))
    x = Variable(x_data)
    h=F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv1(x))),3,stride=2)
    h=F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv2(h))),3,stride=2)
    h=F.relu(model.conv3(h))
    h=F.relu(model.conv4(h))
    h=F.max_pooling_2d(F.relu(model.conv5(h)),3,stride=2)
    h=F.relu(model.fc6(h))
    y = model.fc7(h)

    y_f=F.sigmoid(y)
    return y_f


def predict_labels(model, image_path, mode = 1):
    p_labels = np.zeros(260)

    xdata = load_image(image_path)

    y_f = predict(model, xdata)

    label_prob = y_f.data[0, :]

    limit = 0
    idxsort = label_prob.argsort()
    for i in range(len(idxsort)):
        i = -i - 1
        prob = label_prob[idxsort[i]]

        if mode == 1:
            if prob > 0.1:
                p_labels[idxsort[i]] = 1
        else:
            p_labels[idxsort[i]] = 1
            limit = limit + 1
            if limit > 5:
                break

        #if prob > 0.1:
            #p_labels[idxsort[i]] = 1

    return p_labels.astype(int)

# In[247]:

test_co, test_idx = gen_test_co()

def calc_f1_score(model, mode=1):
    total_score = 0
    inum = 0
    for idir in test_co.keys():
        i = 0
        for iid in test_co[idir]:
            val = []
            i += 1
            #if i > 2:
                #break

            inum += 1
            if inum == 1 or inum % 100 == 0:
                print("calc......", inum)

            idx = get_test_img_idx(test_idx, idir, iid)
            path = '%s/%s.jpeg' % (idir, iid)

            gt_labels = get_origin_labels(idir, iid)

            #gt_labels = get_gt_labels(inum - 1)

            p_labels = predict_labels(model, path, mode)

            f1_score = sklearn.metrics.f1_score(gt_labels, p_labels)

            total_score += f1_score

    return total_score / inum

f1_score = calc_f1_score(without_model, 1)
print(f1_score)
#f1_score = calc_f1_score(word2vec_model, 1)
#print(f1_score)



