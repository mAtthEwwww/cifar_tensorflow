import argparse
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

import resnet
import cifar10_input

NUM_SAMPLES = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TEST

def main(_):
    with tf.Graph().as_default() as g:
        filenames = cifar10_input.get_filenames(
                data_dir=FLAGS.dataset_dir,
                isTrain=False)

        images, labels = cifar10_input.load_batch(
                filenames=filenames,
                batch_size=FLAGS.batch_size,
                isTrain=False,
                isShuffle=False)

        logits = resnet.inference(images)

        correct_prediction = tf.nn.in_top_k(
                predictions=logits,
                targets=labels,
                k=1)

        variable_averages = tf.train.ExponentialMovingAverage(
                decay=FLAGS.variable_averages_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.test_dir, g)

        with tf.Session() as sess:
            checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)

                global_step = checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else :
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()

            try :
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(
                        sess=sess,
                        coord=coord,
                        daemon=True,
                        start=True))

                num_iter = int(np.ceil(NUM_SAMPLES / FLAGS.batch_size))
                true_counter = 0
                total_samples = num_iter * FLAGS.batch_size
                step = 0
                while step < num_iter and not coord.should_stop():
                    num_correct = sess.run(correct_prediction)
                    true_counter += np.sum(num_correct)
                    step += 1

                precision = true_counter / total_samples
                print('%s: precision :%.3f' % (datetime.now(), precision))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision', simple_value=precision)
                summary_writer.add_summary(summary, global_step)

            except Exception as e:
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--dataset_dir',
            type=str,
            default='/home/mattheww/machine_learning/datasets/cifar10/cifar-10-batches-bin',
            help='Directory where to load the dataset.')
    parser.add_argument(
            '--train_dir',
            type=str,
            default='/home/mattheww/machine_learning/datasets/cifar10/resnet_train_log',
            help='Director where to load checkpoint')
    parser.add_argument(
            '--test_dir',
            type=str,
            default='/home/mattheww/machine_learning/datasets/cifar10/resnet_test_log',
            help='Directory where to write eval logs')
    parser.add_argument(
            '--log_device_placement',
            type=bool,
            default=False,
            help='Whether to log device placement.')
    parser.add_argument(
            '--max_steps',
            type=int,
            #default=20000,
            default=20000,
            help='Number of batches to run.')
    parser.add_argument(
            '--batch_size',
            type=int,
            default=64,
            help='Number of images to process in a batch.')
    parser.add_argument(
            '--log_frequency',
            type=int,
            default=10,
            help='How often to log results to the console.')
    parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.05,
            help='Initial Learning rate')
    parser.add_argument(
            '--decay_epochs',
            type=float,
            default=5.0,
            help='Number of epochs per learning rate decay.')
    parser.add_argument(
            '--decay_factor',
            default=0.1,
            help='Factor of learning rate decay.')
    parser.add_argument(
            '--variable_averages_decay',
            type=float,
            default=0.99,
            help='Exponential moving average factor of variables.')


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

