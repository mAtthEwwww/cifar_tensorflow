import argparse
import os
import sys
import time
from datetime import datetime

import tensorflow as tf
import cifar10_input
import resnet

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


STAGE = [32000, 48000]

def _grad_summary(gradpairs):
    for grad, var in gradpairs:
        if grad is not None :
            tf.summary.histogram(var.op.name + '/gradient', grad)

def _variable_summary():
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + '/value', var)

def _lr_in_stage(learning_rate, step, stage):
    if step < FLAGS.warming_step:
        return FLAGS.warming_learning_rate
    for i in range(len(stage)):
        if step > stage[i]:
            learning_rate *= FLAGS.decay_factor
        else :
            break
    return learning_rate


def train(total_loss, global_step, lr):

    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.MomentumOptimizer(lr, 0.9)

    gradpairs = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(gradpairs, global_step=global_step)
        
    tf.summary.scalar('learning_rate', lr)

    _variable_summary()

    _grad_summary(gradpairs)

    variable_averages = tf.train.ExponentialMovingAverage(
            decay=FLAGS.variable_averages_decay,
            num_updates=global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def main(_):
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        filenames = cifar10_input.get_filenames(
                                    data_dir=FLAGS.dataset_dir, 
                                    isTrain=True)
        
        images, labels = cifar10_input.load_batch(
                            filenames=filenames,
                            batch_size=FLAGS.batch_size,
                            isTrain=True,
                            isShuffle=True)

        logits = resnet.inference(images, isTrain=True)

        total_loss = resnet.loss(logits, labels)

        lr = tf.placeholder(shape=[], dtype=tf.float32)

        train_op = train(total_loss, global_step, lr)

        class _Hook(tf.train.SessionRunHook):
        
            def begin(self):
                self._lr = FLAGS.learning_rate
                self._start_time = time.time()
        
            def before_run(self, run_context):
                return tf.train.SessionRunArgs(
                        [total_loss, global_step],
                        feed_dict={lr: self._lr})

            def after_run(self, run_context, run_values):
                step = run_values.results[1] - 1

                self._lr = _lr_in_stage(FLAGS.learning_rate, step, STAGE)

                if step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results[0]

                    examples_per_sec = FLAGS.log_frequency*FLAGS.batch_size/duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)
                    format_str = ('lr: %.4f: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (self._lr, step, loss_value,
                                         examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir, 
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                    tf.train.NanTensorHook(total_loss),
                    _Hook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


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
            default='/home/mattheww/machine_learning/datasets/cifar10/resnet_train',
            help='Directory where to write event logs')
    parser.add_argument(
            '--log_device_placement',
            type=bool,
            default=False,
            help='Whether to log device placement.')
    parser.add_argument(
            '--max_steps',
            type=int,
            default=80000,
            help='Number of batches to run.')
    parser.add_argument(
            '--warming_learning_rate',
            type=float,
            default=0.01,
            help='Warming learning rate',)
    parser.add_argument(
            '--warming_step',
            type=int,
            default=500,
            help='Number of step for warming up')
    parser.add_argument(
            '--batch_size',
            type=int,
            default=128,
            help='Number of images to process in a batch.')
    parser.add_argument(
            '--log_frequency',
            type=int,
            default=10,
            help='How often to log results to the console.')
    parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.1,
            help='Initial Learning rate')
    parser.add_argument(
            '--decay_epochs',
            type=float,
            default=20.0,
            help='Number of epochs per learning rate decay.')
    parser.add_argument(
            '--decay_factor',
            type=float,
            default=0.1,
            help='Factor of learning rate decay.')
    parser.add_argument(
            '--variable_averages_decay',
            type=float,
            default=0.99,
            help='Exponential moving average factor of variables.')
        

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

