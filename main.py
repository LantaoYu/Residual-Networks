import tensorflow as tf
import os
import cPickle
import argparse
import sys
from data_loader import CIFAR_loader
from resnet import ResNet
import numpy as np


def test(sess, resnet, dataloader):
    pred_Y = sess.run(resnet.output, {resnet.images: dataloader.testX})
    pred_Y = np.argmax(pred_Y, 1)
    correct_num = 0
    for i, j in zip(pred_Y, dataloader.testY):
        if i == j:
            correct_num += 1
    print 1.0 * correct_num / len(dataloader.testY)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--num_units', type=str, default='3,3,3')
    argparser.add_argument('--filter_size', type=str, default='16,32,64')
    argparser.add_argument('--stride', type=str, default='1,1,1')
    argparser.add_argument('--class_num', type=int, default=10)
    argparser.add_argument('--learning_rate', type=float, default=0.001)
    argparser.add_argument('--data_dir', type=str, default='cifar-10-batches-py')
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--epoch_num', type=int, default=100)
    argparser.add_argument('--evaluate_every_epoch', type=int, default=1)
    params = argparser.parse_args()

    params.num_units = [int(i) for i in params.num_units.split(',')]
    params.filter_size = [int(i) for i in params.filter_size.split(',')]
    params.stride = [int(i) for i in params.stride.split(',')]

    dataloader = CIFAR_loader(params)
    resnet = ResNet(params)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for e in xrange(params.epoch_num):
        dataloader.reset_counter()
        for batch in xrange(dataloader.num_batches):
            batch_x, batch_y = dataloader.next_batch()
            sess.run(resnet.train_op, {resnet.images: batch_x, resnet.labels: batch_y})
        if e % params.evaluate_every_epoch == 0:
            test(sess, resnet, dataloader)
