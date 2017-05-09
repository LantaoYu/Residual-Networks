import cPickle
import os
import numpy as np


class CIFAR_loader(object):
    def __init__(self, params):
        self.params = params

        train_files = [name for name in os.listdir(self.params.data_dir) if 'data_batch' in name]
        test_files = [name for name in os.listdir(self.params.data_dir) if 'test_batch' in name]

        self.trainX = []
        self.trainY = []
        for file in train_files:
            with open(os.path.join(self.params.data_dir, file), 'rb') as f:
                datadict = cPickle.load(f)
                X = datadict['data']
                self.trainX.append(X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float"))
                self.trainY.append(np.array(datadict['labels']))
        self.trainX = np.concatenate(self.trainX, 0)
        self.trainY = np.concatenate(self.trainY, 0)

        for file in test_files:
            with open(os.path.join(self.params.data_dir, file), 'rb')as f:
                datadict = cPickle.load(f)
                X = datadict['data']
                self.testX = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
                self.testY = np.array(datadict['labels'])

        # shuffle the train data
        new_index = np.random.permutation(len(self.trainX))
        self.trainX = self.trainX[new_index]
        self.trainY = self.trainY[new_index]

        self.data_size = len(self.trainY)
        if self.data_size % self.params.batch_size == 0:
            self.num_batches = self.data_size / self.params.batch_size
        else:
            self.num_batches = self.data_size / self.params.batch_size + 1

        # Change label to one_hot
        self.trainY = np.array([self.one_hot(i, self.params.class_num) for i in self.trainY])
        self.counter = 0

    def one_hot(self, x, out_dims):
        ret = np.zeros(out_dims)
        ret[x] = 1
        return ret

    def next_batch(self):
        next_batch_X = self.trainX[self.counter:self.counter + self.params.batch_size]
        next_batch_Y = self.trainY[self.counter:self.counter + self.params.batch_size]
        self.counter = (self.counter + self.params.batch_size) % self.data_size
        return next_batch_X, next_batch_Y

    def reset_counter(self):
        self.counter = 0
