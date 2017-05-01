import tensorflow as tf


def resnet(inpt, n):
    if n < 20 or (n - 20) % 12 != 0:
        print "ResNet depth invalid."
        return

    num_conv = (n - 20) / 12 + 1
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [3, 3, 3, 16], 1)
        layers.append(conv1)

    for i in range(num_conv):
        with tf.variable_scope('conv2_%d' % (i + 1)):
            conv2_x = residual_block(layers[-1], 16, False)
            conv2 = residual_block(conv2_x, 16, False)
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.get_shape().as_list()[1:] == [32, 32, 16]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i + 1)):
            conv3_x = residual_block(layers[-1], 32, down_sample)
            conv3 = residual_block(conv3_x, 32, False)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [16, 16, 32]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i + 1)):
            conv4_x = residual_block(layers[-1], 64, down_sample)
            conv4 = residual_block(conv4_x, 64, False)
            layers.append(conv4_x)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        assert global_pool.get_shape().as_list()[1:] == [64]

        out = softmax_layer(global_pool, [64, 10])
        layers.append(out)

    return layers[-1]


class ResNets():
    def __init__(self, params):
        """
        :param params:
         params.batch_size
         params.num_units: a list, number of residual units for each stage
         params.filter_size: a list, number of filters for each stage
         params.stride: a list, stride for each stage
        """
        self.params = params

        self.images = tf.placeholder(tf.float32, [self.params.batch_size, 32, 32, 3])
        self.labels = tf.placeholder(tf.float32, [self.params.batch_size, 10])

        with tf.variable_scope('init'):
            x = self._conv(self.images, [3, 3, 3, 16], 1)

        for i in xrange(len(self.params.filter_size)):
            for j in xrange(self.params.num_units):
                with tf.variable_scope('unit%d_%d' % (i, j)):
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

    def _conv(self, input, filter_shape, stride):
        """Convolutional layer"""
        return tf.nn.conv2d(input, filter=self.init_tensor(filter_shape), strides=[1, stride, stride, 1],
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
        option 1: zero padding
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
            else:
                print "Not implemented error"
                exit(1)
        else:
            identity = input_
        return x + identity

    def init_tensor(self, shape):
        return tf.truncated_normal(shape, mean=0.0, stddev=1.0)


def residual_block(inpt, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1, 2, 2, 1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer
    return res


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h


def conv_layer(input, filter_shape, stride):
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(input, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")

    mean, var = tf.nn.moments(conv, axes=list(range(len(conv.get_shape()) - 1)))
    out_channels = filter_shape[3]
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)

    return out
