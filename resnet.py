import numpy as np
import tensorflow as tf
import cifar10_input

NUM_CHANNELS = cifar_input.CHANNEL
NUM_CLASSES = cifar10_input.NUM_CLASSES

CONV_WEIGHT_DECAY = 0.00005
FC_WEIGHT_DECAY = 0.00005
#Tensor.get_shape().as_list()

def _conv(name, x, kernel_shape, strides, weight_decay=0.0):
    with tf.variable_scope(name):
        with tf.device('/cpu:0'):
            if weight_decay > 0 :
                regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
            else :
                regularizer = None

            fan_in = kernel_shape[0] * kernel_shape[1] * kernel_shape[2]

            kernel = tf.get_variable(
                    name='weights',
                    shape=kernel_shape,
                    dtype=float32,
                    initializer=tf.random_normal_initializer(
                        stddev=np.sqrt(2.0/fan_in),
                        seed=None,
                        dtype=tf.float32),
                    regularizer=regularizer)

        return tf.nn.conv2d(
                input=x,
                filter=kernel,
                strides=strides,
                padding='SAME')

def _fully_connected(name, x, out_dim, weight_decay=0.0):
    with tf.variable_scope(name):
        with tf.device('/cpu:0'):
            if weight_decay > 0 :
                regularizer = tf.contrib.layer.l2_regularizer(weight_decay)
            else :
                regularizer = None

            batch_size = x.get_shape()[0].value()

            x = tf.reshape(x, [batch_size, -1])

            W = tf.get_variable(
                    name='weights',
                    shape=[x.get_shape()[1].value, out_dim],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer(
                        stddev=np.sqrt(1.0/x.get_shape()[0].value),
                        seed=None,
                        dtype=tf.float32),
                    regularizer=regularizer)

            b = tf.get_varible(
                    name='biases',
                    shape=[out_dim],
                    dtype=tf.float32,
                    initializer=tf.random_constant_initializer(
                        value=0.0
                        dtype=tf.float32))

        return tf.bias_add(tf.matmul(x, W), b)

def _batch_norm(name, x, isTrain):
    with tf.variable_scope(name):
        with tf.device('/cpu:0'):
            num_channels = x.get_shape()[-1].value

            gamma = tf.get_variable(
                    name='gamma',
                    shape=[num_channels],
                    initializer=tf.constant_initializer(
                        value=1.0,
                        dtype=tf.float32))

            beta = tf.get_variable(
                    name='beta',
                    shape=[num_channels],
                    initializer=tf.constant_initializer(
                        value=0.0,
                        dtype=tf.float32))

            moving_mean = tf.get_variable(
                    name='moving_mean',
                    shape=[num_channels],
                    initializer=tf.constant_initializer(
                        value=0.0,
                        dtype=tf.float32),
                    trainable=False)

            moving_variance = tf.get_variable(
                    name='moving_variance',
                    initializer=tf.constant_initializer(
                        value=1.0,
                        dtype=tf.float32),
                    trainable=False)

        if isTrain:
            axes = list(range(len(x.get_shape) - 1))

            mean, variance = tf.moments(
                    x=x,
                    axes=axes,
                    name='moments')

            ema = tf.train.ExponentialMovingAverage(0.99, name='avg')

            ema_op = ema.apply([mean, variance])

            with tf.control_dependencies([ema_op]):

                return tf.nn.batch_normaliztion(
                        x=x,
                        mean=mean,
                        variance=variance,
                        offset=beta,
                        scale=gamma,
                        variance_epsilon=0.0001)

        else :
            return tf.nn.batch_normalization(
                    x=x,
                    mean=moving_mean,
                    variance=moving_variance,
                    offset=beta,
                    scale=gamma,
                    variance_epsilon=0.0001)


def _residual(x, in_filter, out_filter, strides, isTrain):
    orig_x = x

    with tf.variable_scope('sub1'):
        x = _batch_norm(
                name='bn1', 
                x=x, 
                isTrain=isTrain)

        x = tf.nn.relu(x)

        x = _conv(
                name='conv1',
                x=x,
                kernel_shape=[3, 3, in_filter, out_filter],
                strides=strides)

    with tf.variable_scope('sub2'):
        x = _batch_norm(
                name='bn2',
                x=x,
                isTrain=isTrain)

        x = tf.nn.relu(x)

        x = _conv(
                name='conv2',
                x=x,
                kernel_shape=[3, 3, out_filter, out_filter],
                strides=[1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
            x = _conv(
                    name='conv3',
                    x=x,
                    kernel_shape=[1, 1, in_filter, out_filter],
                    strides=strides)

        else if strides != [1, 1, 1, 1]:
            orig_x = tf.nn.avg_pool(
                    value=orig_x, 
                    ksize=strides,
                    strides=strides,
                    padding='VALID')
        x += orig_x

    return x

def _global_average_pool(x):
    return tf.reduce_mean(x, [1, 2])

def inference(
        image,
        num_blocks=[5, 5, 5],
        num_channels=[16, 16, 32, 64],
        isTrain=False): 

    num_blocks = [1] + num_blocks
    sampling_strides = [1, 2, 2, 1]
    unsampling_strides = [1, 1, 1, 1]

    with tf.variable_scope('part_0'):
        x = _conv(
                name='conv',
                x=image, 
                kernel_shape=[3, 3, NUM_CHANNELS, num_channels[0]],
                strides=umsampling_strides,
                weight_decay=CONV_WEIGHT_DECAY)

    for part in range(1, len(num_channels)):
        with tf.variable_scope('part_%d' % part):
            with tf.variable_scope('block_0'):
                x = _residual(
                        x=x,
                        in_filter=num_channels[part-1], 
                        out_filter=num_channels[part],
                        strides=sampling_strides,
                        isTrain=isTrain)

            for block in range(1, num_blocks[part]):
                with tf.variable_scope('block_%d' % i):
                    x = _residual(
                            x=x,
                            in_filter=num_channels[part],
                            out_filter=num_channels[part],
                            strides=unsampling_strides,
                            isTrain=isTrain)

    with tf.variable_scope('part_last'):
        x = _batch_norm(
                name='bn_out',
                x=x,
                isTrain=isTrain)
        x = tf.nn.relu(x)

        x = _global_avg_pool(x)

    with tf.variable_scope('logits'):
        logits = _fully_connected(x, NUM_CLASSES)

    return logits

def loss(logits, labels):
    sample_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
            name='sample_cross_entropy')

    batch_cross_entropy = tf.reduce_mean(
            input_tensor=batch_cross_entropy,
            axis=0,
            name='cross_entropy')

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([batch_cross_entropy] + regularization_losses)

    tf.scalar_summary('loss', loss_)

    return loss_
