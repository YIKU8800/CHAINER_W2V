from PIL import Image
from chainer import Variable, optimizers, serializers, cuda
import chainer.functions  as F
import chainer.links as CL
import numpy as np

# np.set_printoptions(threshold=np.inf)
np.random.seed(0)
import cupy

cupy.random.seed(0)
import scipy.io
import time
import pickle
import chainer
import gc
import csv

PICKLE_PATH = "corel5k/alex_net.pkl"

rate = 1.0
weight_p = 1.0
dim = 160

n_epoch = 10
batchsize = 100

L = np.load('./L/L_%.2f/L_%d.npy' % (weight_p, dim))
L = L[np.newaxis, :].astype(np.float32)
L_origin = L

for i in range(batchsize - 1):
    L = np.concatenate((L, L_origin), axis=0)

# build and read the model

model = chainer.Chain(conv1=CL.Convolution2D(3,  96, 11, stride=4),
                    bn1=CL.BatchNormalization(96),
                    conv2=CL.Convolution2D(96, 256,  5, pad=2),
                    bn2=CL.BatchNormalization(256),
                    conv3=CL.Convolution2D(256, 384,  3, pad=1),
                    conv4=CL.Convolution2D(384, 384,  3, pad=1),
                    conv5=CL.Convolution2D(384, 256,  3, pad=1),
                    fc6=CL.Linear(2304,1024),
                    fc7=CL.Linear(1024, 260))

## copy parameter
with open(PICKLE_PATH, "rb") as pkl_file:
    original_model = pickle.load(pkl_file)

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
    # model.fc6.W.data = original_model.fc6.W.data
    # model.fc6.b.data = original_model.fc6.b.data
    # model.fc7.W.data = original_model.fc7.W.data
    # model.fc7.b.data = original_model.fc7.b.data

    del original_model
    gc.collect()

model.to_gpu()

# make train dataset
im_size = 127
num_label = 260

# load data
x_train = pickle.load(open("corel5k/x_train.pkl", "rb"))
y_train = np.load(open("corel5k/y_train.npy", "rb"))

x_test = pickle.load(open("corel5k/x_test.pkl", "rb"))
y_test = np.load(open("corel5k/y_test.npy", "rb"))

print("load complete.............", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

N = len(x_train)
N_test = len(x_test)


# define forward process
def forward(x_data, y_data, L=L, batchsize=batchsize):
    L = Variable(cuda.to_gpu(L))
    x, t = Variable(cuda.to_gpu(x_data)), Variable(cuda.to_gpu(y_data))
    #    x, t = Variable(x_data), Variable(y_data)
    h = F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv1(x))), 3, stride=2)
    h = F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv2(h))), 3, stride=2)
    h = F.relu(model.conv3(h))
    h = F.relu(model.conv4(h))
    h = F.max_pooling_2d(F.relu(model.conv5(h)), 3, stride=2)
    h = F.relu(model.fc6(h))
    y = model.fc7(h)

    y_f = F.sigmoid(y)

    y_ft = F.expand_dims(y_f, 2)
    term = (F.sum(F.batch_matmul(F.batch_matmul(y_f, L, transa=True), y_ft))) / batchsize

    sce = F.sigmoid_cross_entropy(y, t)
    E=sce+(rate*term)
    #E = sce
    return E, sce, term, y_f


def cul_acc(x_data, y_data, threshold=0.500):
    output = np.array(x_data).astype(np.float32)

    output = output.reshape(len(y_data) * 260, )
    target = y_data.reshape(len(y_data) * 260, )
    output[np.where(output > threshold)[0]] = 1
    output[np.where(output < threshold)[0]] = 0
    correct = np.count_nonzero(output == target)
    return float(correct) / (len(y_data) * 260)


# setup optimizer
optimizer = optimizers.Adam(alpha=0.001)
#optimizer = optimizers.SGD(lr=0.1)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.001))

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_loss = 0
    sum_sce = 0
    sum_term = 0
    loss_val = []
    for i in range(0, N, batchsize):
        x_batch = x_train[perm[i:i + batchsize]]
        y_batch = y_train[perm[i:i + batchsize]]

        # optimizer.zero_grads()
        loss, sce, GFHF, pred = forward(x_batch, y_batch)
        #print("fffffffffffffffffffffff", loss, sce, pred)
        loss.backward()
        #optimizer.update()
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_sce += float(cuda.to_cpu(sce.data)) * batchsize
        sum_term += float(cuda.to_cpu(GFHF.data)) * batchsize
        num_train = i + batchsize
    if (num_train % 500) == 0:
        print('num_train=', num_train, 'loss=', sum_loss / num_train)

    print('train loss={},sce={},term={}'.format(sum_loss / N, sum_sce / N, sum_term / N))

    # test for training dataset
    sum_loss = 0
    sum_sce = 0
    sum_term = 0
    sum_acc = 0
    loss_val = []
    for i in range(0, N, batchsize):
        x_batch = x_train[perm[i:i + batchsize]]
        y_batch = y_train[perm[i:i + batchsize]]

        loss, sce, GFHF, pred = forward(x_batch, y_batch)
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_sce += float(cuda.to_cpu(sce.data)) * batchsize
        sum_term += float(cuda.to_cpu(GFHF.data)) * batchsize
        pred = cuda.to_cpu(pred.data)
    #        print np.max(pred)
    acc = cul_acc(pred, y_batch)
    sum_acc += acc * batchsize
    print('test train loss={},sce={},term={},acc={}'.format(sum_loss / N, sum_sce / N, sum_term / N, sum_acc / N))
    #    loss_val.append((sum_sce+sum_term)/N)
    loss_val.append(sum_loss / N)
    loss_val.append(sum_sce / N)
    loss_val.append(sum_term / N)
    loss_val.append((1.0 - (sum_acc / N)))
    """"
    with open('./loss_data/sgd/eigenvector/train_%.6f_%.2f_%d.csv'%(rate,weight_p,dim),'ab') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(loss_val)
    """
    # test for training dataset
    sum_loss = 0
    sum_sce = 0
    sum_term = 0
    sum_acc = 0
    loss_val = []
    for i in range(0, N_test):
        x_batch = x_test[i:i + 1]
        y_batch = y_test[i:i + 1]

        loss, sce, GFHF, pred = forward(x_batch, y_batch, L=L_origin, batchsize=1)
        sum_loss += float(cuda.to_cpu(loss.data))
        sum_sce += float(cuda.to_cpu(sce.data))
        sum_term += float(cuda.to_cpu(GFHF.data))
    pred = cuda.to_cpu(pred.data)

    acc = cul_acc(pred, y_batch)
    sum_acc += acc
    print('test test loss={},sce={},term={},acc={}'.format(sum_loss / N_test, sum_sce / N_test, sum_term / N_test,
                                                           sum_acc / N_test))
    #    loss_val.append((sum_sce+sum_term)/N)
    loss_val.append(sum_loss / N)
    loss_val.append(sum_sce / N_test)
    loss_val.append(sum_term / N_test)
    loss_val.append(1.0 - (sum_acc / N_test))
    """
    with open('./loss_data/sgd/eigenvector/test_%.6f_%.2f_%d.csv'%(rate,weight_p,dim),'ab') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(loss_val)
    """
model.to_cpu()
# serializers.save_npz('./model_%.6f_%.2f_%d.model'%(rate,weight_p,dim),model)

print("saving model......................")
pickle.dump(model, open(PICKLE_PATH, "wb"))