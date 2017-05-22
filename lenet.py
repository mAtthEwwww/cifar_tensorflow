import tensorflow as tf
import cifar10_input

NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_CHANNELS = cifar10_input.CHANNEL


"""
Network abstraction
"""
conv1_weights_shape = [5, 5, NUM_CHANNELS, 64]
conv1_weights_strides = [1, 1, 1, 1]
conv1_weights_padding = 'SAME'
conv1_weights_decay = 0.0
conv1_weights_initializer = tf.truncated_normal_initializer(
                                    stddev=5e-2,
                                    seed=None,
                                    dtype=tf.float32)
conv1_biases_shape = [64]
conv1_biases_initializer = tf.constant_initializer(0.0)

pool1_shape = [1, 3, 3, 1]
pool1_strides = [1, 2, 2, 1]
pool1_padding = 'SAME'

conv2_weights_shape = [5, 5, 64, 64]
conv2_weights_strides = [1, 1, 1, 1]
conv2_weights_padding = 'SAME'
conv2_weights_decay = 0.0
conv2_weights_initializer=tf.truncated_normal_initializer(
                                    stddev=5e-2,
                                    seed=None,
                                    dtype=tf.float32)
conv2_biases_shape = [64]
conv2_biases_initializer = tf.constant_initializer(0.0)

pool2_shape = [1, 3, 3, 1]
pool2_strides = [1, 2, 2, 1]
pool2_padding = 'SAME'

fn3_num_units = 384
fn3_biases_initializer = tf.constant_initializer(0.1)
fn3_weights_decay = 0.004
fn3_weights_initializer = tf.truncated_normal_initializer(
                                    stddev=0.04,
                                    seed=None,
                                    dtype=tf.float32)

fn4_num_units = 192
fn4_biases_initializer = tf.constant_initializer(0.1)
fn4_weights_decay = 0.004
fn4_weights_initializer = tf.truncated_normal_initializer(
                                    stddev=0.04,
                                    seed=None,
                                    dtype=tf.float32)

fn5_num_units = NUM_CLASSES
fn5_biases_initializer = tf.constant_initializer(0.1)
fn5_weights_decay = 0.0
fn5_weights_initializer = tf.truncated_normal_initializer(
                                    stddev=1/192.0,
                                    seed=None,
                                    dtype=tf.float32)


def inference(images, isTrain=False):

    l2_losses = [];

    with tf.variable_scope('conv1') as scope:
        with tf.device('/cpu:0'):
            kernel = tf.get_variable(
                    name='weights',
                    shape=conv1_weights_shape,
                    dtype=tf.float32,
                    initializer=conv1_weights_initializer)

            biases = tf.get_variable(
                    name='biases',
                    shape=conv1_biases_shape,
                    initializer=conv1_biases_initializer)

            l2_loss = tf.multiply(
                    tf.nn.l2_loss(kernel),
                    conv2_weights_decay,
                    name='weight_loss')

            l2_losses.append(l2_loss)

        conv1 = tf.nn.conv2d(
                input=images,
                filter=kernel,
                strides=conv1_weights_strides,
                padding=conv1_weights_padding)

        conv1 = tf.nn.bias_add(conv1, biases)

        conv1 = tf.nn.relu(conv1)

        tf.summary.histogram('/activations', conv1)

        tf.summary.scalar('/sparsity', tf.nn.zero_fraction(conv1))

    pool1 = tf.nn.max_pool(
            value=conv1,
            ksize=pool1_shape,
            strides=pool1_strides,
            padding=pool1_padding)

    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    
    with tf.variable_scope('conv2') as scope:
        with tf.device('/cpu:0'):
            kernel = tf.get_variable(
                    name='weights',
                    shape=conv2_weights_shape,
                    dtype=tf.float32,
                    initializer=conv2_weights_initializer)

            biases = tf.get_variable(
                    name='biases',
                    shape=conv2_biases_shape,
                    dtype=tf.float32)

            l2_loss = tf.multiply(
                    tf.nn.l2_loss(kernel),
                    conv2_weights_decay,
                    name='weight_loss')

            l2_losses.append(l2_loss)

        conv2 = tf.nn.conv2d(
                input=norm1,
                filter=kernel,
                strides=conv2_weights_strides,
                padding=conv2_weights_padding)

        conv2 = tf.nn.bias_add(conv2, biases)

        conv2 = tf.nn.relu(conv2)

        tf.summary.histogram('/activations', conv2)

        tf.summary.scalar('/sparsity', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    pool2 = tf.nn.max_pool(
            value=norm2,
            ksize=pool2_shape,
            strides=pool2_strides,
            padding=pool2_padding)

    with tf.variable_scope('fn3') as scope:
        shape = pool2.get_shape()
        flatten3 = tf.reshape(
                pool2,
                [-1, shape[1].value * shape[2].value * shape[3].value])

        with tf.device('/cpu:0'):
            weights = tf.get_variable(
                    name='weights',
                    shape=[flatten3.get_shape()[1].value, fn3_num_units],
                    dtype=tf.float32,
                    initializer=fn3_weights_initializer)

            biases = tf.get_variable(
                    name='biases',
                    shape=[fn3_num_units],
                    dtype=tf.float32,
                    initializer=fn3_biases_initializer)

            l2_loss = tf.multiply(
                    tf.nn.l2_loss(weights),
                    fn3_weights_decay,
                    name='weight_loss')

            l2_losses.append(l2_loss)

        fn3 = tf.matmul(flatten3, weights)

        fn3 = tf.nn.bias_add(fn3, biases)
        
        fn3 = tf.nn.relu(fn3)

        tf.summary.histogram('/activation', fn3)

        tf.summary.scalar('/sparsity', tf.nn.zero_fraction(fn3))

    with tf.variable_scope('fn4') as scope:
        with tf.device('/cpu:0'):
            weights = tf.get_variable(
                    name='weights',
                    shape=[fn3_num_units, fn4_num_units],
                    dtype=tf.float32,
                    initializer=fn4_weights_initializer)

            biases = tf.get_variable(
                    name='biases',
                    shape=[fn4_num_units],
                    dtype=tf.float32,
                    initializer=fn4_biases_initializer)

            l2_loss = tf.multiply(
                    tf.nn.l2_loss(weights),
                    fn4_weights_decay,
                    name='weight_loss')

            l2_losses.append(l2_loss)

        fn4 = tf.matmul(fn3, weights)

        fn4 = tf.nn.bias_add(fn4, biases)
        
        fn4 = tf.nn.relu(fn4)

        tf.summary.histogram('/activation', fn4)

        tf.summary.scalar('/sparsity', tf.nn.zero_fraction(fn4))

    with tf.variable_scope('fn5') as scope:
        with tf.device('/cpu:0'):
            weights = tf.get_variable(
                    name='weights',
                    shape=[fn4_num_units, fn5_num_units],
                    initializer=fn5_weights_initializer)

            biases = tf.get_variable(
                    name='biases',
                    shape=[NUM_CLASSES],
                    dtype=tf.float32,
                    initializer=fn5_biases_initializer)

            l2_loss = tf.multiply(
                    tf.nn.l2_loss(weights),
                    fn5_weights_decay,
                    name='weight_loss')

            l2_losses.append(l2_loss)

        fn5 = tf.matmul(fn4, weights)

        fn5 = tf.nn.bias_add(fn5, biases)

        tf.summary.histogram('/activations', fn5)

        tf.summary.scalar('/sparsity', tf.nn.zero_fraction(fn5))
        
    logits = fn5

    if isTrain:
        return logits, l2_losses
    else :
        return logits


def loss(logits, labels, losses):
    tf.cast(labels, tf.float32)
    batch_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
            name='cross_entropy')
    mean_cross_entropy = tf.reduce_mean(
                            batch_cross_entropy,
                            name='loss/cross_entropy',
                            axis=0)

    losses.append(mean_cross_entropy)

    total_loss = tf.add_n(losses, name='loss/total_loss')

    losses.append(total_loss)

    for l in losses:
        tf.summary.scalar(l.op.name, l)

    ema = tf.train.ExponentialMovingAverage(0.9, name='avg')

    loss_ema_op = ema.apply([total_loss])

    tf.summary.scalar(total_loss.op.name + '_average', ema.average(total_loss))

    with tf.control_dependencies([loss_ema_op]):
        # make sure loss_ema_op is running
        total_loss = tf.identity(total_loss)

    return total_loss



