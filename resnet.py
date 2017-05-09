import tensorflow as tf


class ResNet(object):
    def __init__(self, params):
        self.params = params

        self.images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.labels = tf.placeholder(tf.float32, [None, 10])

        with tf.variable_scope('init'):
            x = self._conv(self.images, [3, 3, 3, 16], 1)

        for i in xrange(len(self.params.filter_size)):
            for j in xrange(len(self.params.num_units)):
                with tf.variable_scope('unit_%d_sublayer_%d' % (i, j)):
                    if j == 0:
                        if i == 0:
                            # transition from init stage to the first stage stage
                            x = self._residual_unit(x, 16, self.params.filter_size[i], self.params.stride[i])
                        else:
                            x = self._residual_unit(x, self.params.filter_size[i - 1], self.params.filter_size[i],
                                                    self.params.stride[i])
                    else:
                        x = self._residual_unit(x, self.params.filter_size[i], self.params.filter_size[i],
                                                self.params.stride[i])
        with tf.variable_scope('output'):
            x = self._batch_norm(x)
            x = tf.nn.relu(x)
            x = tf.reduce_mean(x, [1, 2])
            self.output = self._softmax_layer(x, self.params.filter_size[-1], self.params.class_num)

        self.loss = - tf.reduce_sum(self.labels * tf.log(self.output))
        self.optimizer = tf.train.MomentumOptimizer(self.params.learning_rate, 0.9)
        # self.optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def _softmax_layer(self, input_, in_dims, out_dims):
        W = self.init_tensor([in_dims, out_dims])
        b = tf.Variable(tf.zeros([out_dims]))
        return tf.nn.softmax(tf.matmul(input_, W) + b)

    def _conv(self, input, filter_shape, stride):
        """Convolutional layer"""
        return tf.nn.conv2d(input,
                            filter=self.init_tensor(filter_shape),
                            strides=[1, stride, stride, 1],
                            padding="SAME")

    def _batch_norm(self, input_):
        """Batch normalization for a 4-D tensor"""
        assert len(input_.get_shape()) == 4
        filter_shape = input_.get_shape().as_list()
        mean, var = tf.nn.moments(input_, axes=[0, 1, 2])
        out_channels = filter_shape[3]
        offset = tf.Variable(tf.zeros([out_channels]))
        scale = tf.Variable(tf.ones([out_channels]))
        batch_norm = tf.nn.batch_normalization(input_, mean, var, offset, scale, 0.001)
        return batch_norm

    def _residual_unit(self, input_, in_filters, out_filters, stride, option=0):
        """
        Residual unit with 2 sub-layers
        When in_filters != out_filters:
        option 0: zero padding
        """
        # first convolution layer
        x = self._batch_norm(input_)
        x = tf.nn.relu(x)
        x = self._conv(x, [3, 3, in_filters, out_filters], stride)
        # second convolution layer
        x = self._batch_norm(x)
        x = tf.nn.relu(x)
        x = self._conv(x, [3, 3, out_filters, out_filters], stride)

        if in_filters != out_filters:
            if option == 0:
                difference = out_filters - in_filters
                left_pad = difference / 2
                right_pad = difference - left_pad
                identity = tf.pad(input_, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
                return x + identity
            else:
                print "Not implemented error"
                exit(1)
        else:
            return x + input_

    def init_tensor(self, shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1.0))
