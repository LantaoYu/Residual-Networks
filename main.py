import tensorflow as tf
import os
import cPickle
from data_loader import CIFAR_loader

data_dir = 'dataset/cifar-10-batches-py'

dataloader = CIFAR_loader()
dataloader.read_data(data_dir)
