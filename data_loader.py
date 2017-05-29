import cPickle
import os
import numpy as np


class CIFAR_loader(object):
    def __init__(self, params):
        self.params = params

        train_files = [name for name in os.listdir(self.params.data_dir) if 'data_batch' in name]
        test_files = [name for name in os.listdir(self.params.data_dir) if 'test_batch' in name]

        # Read files
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

        # Change label to one_hot
        self.trainY = np.array([self.one_hot(i, self.params.class_num) for i in self.trainY])

        # For split batch
        self.train_counter = 0
        self.test_counter = 0
        self.train_data_size = len(self.trainY)
        self.train_num_batches = int(np.ceil(1.0 * self.train_data_size / self.params.batch_size))
        self.test_data_size = len(self.testY)
        self.test_num_batches = int(np.ceil(1.0 * self.test_data_size / self.params.batch_size))

    def one_hot(self, x, out_dims):
        ret = np.zeros(out_dims)
        ret[x] = 1
        return ret

    def train_next_batch(self):
        next_batch_X = self.trainX[self.train_counter:self.train_counter + self.params.batch_size]
        next_batch_Y = self.trainY[self.train_counter:self.train_counter + self.params.batch_size]
        self.train_counter = (self.train_counter + self.params.batch_size) % self.train_data_size
        return next_batch_X, next_batch_Y

    def test_next_batch(self):
        next_batch_X = self.testX[self.test_counter:self.test_counter + self.params.batch_size]
        next_batch_Y = self.testY[self.test_counter:self.test_counter + self.params.batch_size]
        self.test_counter = (self.test_counter + self.params.batch_size) % self.test_data_size
        return next_batch_X, next_batch_Y

    def reset_counter(self, mode):
        if mode == 'train':
            self.train_counter = 0
        elif mode == 'test':
            self.test_counter = 0
