import os
import tensorflow as tf

HEIGHT = 24
WIDTH = 24
CHANNEL = 3

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 10000
MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.4


def read_sample(filename_queue):

    height = 32
    width = 32
    channel = 3
    image_bytes = height * width * channel

    label_bytes = 1

    sample_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=sample_bytes)

    _, stringBytes = reader.read(filename_queue)

    uint8Bytes = tf.decode_raw(stringBytes, tf.uint8)

    label = tf.cast(
                tf.slice(uint8Bytes,
                        begin=[0],
                        size=[label_bytes]),
                tf.int32)

    image = tf.transpose(
                tf.reshape(
                    tf.slice(uint8Bytes,
                            begin=[label_bytes],
                            size=[image_bytes]),
                    [channel, height, width]),
                [1, 2, 0])

    return image, label


def load_batch(filenames, batch_size, isTrain=False, isShuffle=False):

    filename_queue = tf.train.string_input_producer(filenames)

    image, label = read_sample(filename_queue)

    image = tf.cast(image, tf.float32)

    label.set_shape([1])

    if isTrain:
        image = tf.random_crop(image, [HEIGHT, WIDTH, CHANNEL])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                MIN_FRACTION_OF_EXAMPLES_IN_QUEUE)

    else :
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                    HEIGHT,
                                                    WIDTH)

        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TEST *
                                MIN_FRACTION_OF_EXAMPLES_IN_QUEUE)


    image = tf.image.per_image_standardization(image)


    if isShuffle:
        images, labels = tf.train.shuffle_batch(
                                [image, label],
                                batch_size=batch_size,
                                num_threads=16,
                                capacity=min_queue_examples + 3 * batch_size,
                                min_after_dequeue=min_queue_examples)
    else :
        images, labels = tf.train.batch(
                                [image, label],
                                batch_size=batch_size,
                                num_threads=16,
                                capacity=min_queue_examples + 3 * batch_size)
                                #min_after_dequeue=min_queue_examples)
                                # 不存在, 不必要

    tf.summary.image('images', images)

    labels = tf.reshape(labels, [batch_size])

    return images, labels


def get_filenames(data_dir, isTrain=True):

    if isTrain:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                    for i in range(1, 6)]

    else :
        filenames = [os.path.join(data_dir, 'test_batch.bin')]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    return filenames


