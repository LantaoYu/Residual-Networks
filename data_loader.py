import cPickle
import os
import numpy as np


class CIFAR_loader():
    def read_data(self, data_dir):
        self.data_dir = data_dir
        train_files = [name for name in os.listdir(data_dir) if 'data_batch' in name]
        test_files = [name for name in os.listdir(data_dir) if 'test_batch' in name]

        self.trainX = []
        self.trainY = []
        for file in train_files:
            with open(os.path.join(self.data_dir, file), 'rb') as f:
                datadict = cPickle.load(f)
                X = datadict['data']
                self.trainX.append(X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float"))
                self.trainY.append(np.array(datadict['labels']))
        self.trainX = np.concatenate(self.trainX, 0)
        self.trainY = np.concatenate(self.trainY, 0)

        for file in test_files:
            with open(os.path.join(self.data_dir, file), 'rb')as f:
                datadict = cPickle.load(f)
                X = datadict['data']
                self.testX = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
                self.testY = np.array(datadict['labels'])

        # shuffle the train data
        new_index = np.random.permutation(self.trainX.shape[0])
        self.trainX = self.trainX[new_index]
        self.trainY = self.trainY[new_index]
