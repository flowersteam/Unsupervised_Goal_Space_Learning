#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains ready to train architectures using standard training and stabilized ones.

Author: Alexandre Péré

"""

from network import BaseNetwork
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import h5py
import numpy as np


class VGG16AE(BaseNetwork):
    """
    An autoencoder based on VGG16 architecture and weights.
    """

    def _construct_arch(self, emb_size=10, signal_size=224, lr=1e-3, beta=1., ann_duration=1e4, bn=False):
        with self._tf_graph.as_default():
            # Define Network # =========================================================================================
            with tf.variable_scope('Input'):
                self._net_input = tf.placeholder(tf.float32, shape=[None, signal_size * signal_size * 3], name='Input')
                batch_size = tf.shape(self._net_input)[0]
                net_is_training = tf.placeholder(tf.bool, name="IsTraining")
                net_inp_input = tf.reshape(self._net_input, [batch_size, signal_size, signal_size, 3])
            with tf.variable_scope("Encoder"):
                with tf.variable_scope("ConvLayer_1"):
                    net_enc_w1 = tf.Variable(tf.truncated_normal([3,3,3,64], stddev=0.001), name="Weights")
                    net_enc_b1 = tf.Variable(tf.constant(0., shape=[64]), name="Biases")
                    net_enc_h1 = tf.nn.conv2d(net_inp_input ,net_enc_w1, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h1 = tf.nn.relu(tf.add(net_enc_h1, net_enc_b1), name="Activation")
                    if bn: net_enc_h1 = batch_norm(net_enc_h1, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_2"):
                    net_enc_w2 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.001), name="Weights")
                    net_enc_b2 = tf.Variable(tf.constant(0., shape=[64]), name="Biases")
                    net_enc_h2 = tf.nn.conv2d(net_enc_h1,net_enc_w2, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h2 = tf.nn.relu(tf.add(net_enc_h2, net_enc_b2), name="Activation")
                    if bn: net_enc_h2 = batch_norm(net_enc_h2, is_training = net_is_training)
                with tf.variable_scope("MaxPool_3"):
                    net_enc_h3 = tf.nn.max_pool(net_enc_h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("ConvLayer_4"):
                    net_enc_w4 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.001), name="Weights")
                    net_enc_b4 = tf.Variable(tf.constant(0., shape=[128]), name="Biases")
                    net_enc_h4 = tf.nn.conv2d(net_enc_h3,net_enc_w4, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h4 = tf.nn.relu(tf.add(net_enc_h4, net_enc_b4), name="Activation")
                    if bn: net_enc_h4 = batch_norm(net_enc_h4, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_5"):
                    net_enc_w5 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.001), name="Weights")
                    net_enc_b5 = tf.Variable(tf.constant(0., shape=[128]), name="Biases")
                    net_enc_h5 = tf.nn.conv2d(net_enc_h4,net_enc_w5, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h5 = tf.nn.relu(tf.add(net_enc_h5, net_enc_b5), name="Activation")
                    if bn: net_enc_h5 = batch_norm(net_enc_h5, is_training = net_is_training)
                with tf.variable_scope("MaxPool_6"):
                    net_enc_h6 = tf.nn.max_pool(net_enc_h5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("ConvLayer_7"):
                    net_enc_w7 = tf.Variable(tf.truncated_normal([3,3,128, 256], stddev=0.001), name="Weights")
                    net_enc_b7 = tf.Variable(tf.constant(0., shape=[256]), name="Biases")
                    net_enc_h7 = tf.nn.conv2d(net_enc_h6,net_enc_w7, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h7 = tf.nn.relu(tf.add(net_enc_h7, net_enc_b7), name="Activation")
                    if bn: net_enc_h7 = batch_norm(net_enc_h7, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_8"):
                    net_enc_w8 = tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.001), name="Weights")
                    net_enc_b8 = tf.Variable(tf.constant(0., shape=[256]), name="Biases")
                    net_enc_h8 = tf.nn.conv2d(net_enc_h7, net_enc_w8, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h8 = tf.nn.relu(tf.add(net_enc_h8, net_enc_b8), name="Activation")
                    if bn: net_enc_h8 = batch_norm(net_enc_h8, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_9"):
                    net_enc_w9 = tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.001), name="Weights")
                    net_enc_b9 = tf.Variable(tf.constant(0., shape=[256]), name="Biases")
                    net_enc_h9 = tf.nn.conv2d(net_enc_h8,net_enc_w9, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h9 = tf.nn.relu(tf.add(net_enc_h9, net_enc_b9), name="Activation")
                    if bn: net_enc_h9 = batch_norm(net_enc_h9, is_training = net_is_training)
                with tf.variable_scope("MaxPool_10"):
                    net_enc_h10 = tf.nn.max_pool(net_enc_h9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("ConvLayer_11"):
                    net_enc_w11 = tf.Variable(tf.truncated_normal([3,3,256,512], stddev=0.001), name="Weights")
                    net_enc_b11 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h11 = tf.nn.conv2d(net_enc_h10,net_enc_w11, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h11 = tf.nn.relu(tf.add(net_enc_h11, net_enc_b11), name="Activation")
                    if bn: net_enc_h11 = batch_norm(net_enc_h11, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_12"):
                    net_enc_w12 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001), name="Weights")
                    net_enc_b12 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h12 = tf.nn.conv2d(net_enc_h11, net_enc_w12, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h12 = tf.nn.relu(tf.add(net_enc_h12, net_enc_b12), name="Activation")
                    if bn: net_enc_h12 = batch_norm(net_enc_h12, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_13"):
                    net_enc_w13 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001), name="Weights")
                    net_enc_b13 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h13 = tf.nn.conv2d(net_enc_h12,net_enc_w13, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h13 = tf.nn.relu(tf.add(net_enc_h13, net_enc_b13), name="Activation")
                    if bn: net_enc_h13 = batch_norm(net_enc_h13, is_training = net_is_training)
                with tf.variable_scope("MaxPool_14"):
                    net_enc_h14 = tf.nn.max_pool(net_enc_h13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("ConvLayer_15"):
                    net_enc_w15 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001), name="Weights")
                    net_enc_b15 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h15 = tf.nn.conv2d(net_enc_h14,net_enc_w15, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h15 = tf.nn.relu(tf.add(net_enc_h15, net_enc_b15), name="Activation")
                    if bn: net_enc_h15 = batch_norm(net_enc_h15, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_16"):
                    net_enc_w16 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001), name="Weights")
                    net_enc_b16 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h16 = tf.nn.conv2d(net_enc_h15, net_enc_w16, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h16 = tf.nn.relu(tf.add(net_enc_h16, net_enc_b16), name="Activation")
                    if bn: net_enc_h16 = batch_norm(net_enc_h16, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_17"):
                    net_enc_w17 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001), name="Weights")
                    net_enc_b17 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h17 = tf.nn.conv2d(net_enc_h16,net_enc_w17, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h17 = tf.nn.relu(tf.add(net_enc_h17, net_enc_b17), name="Activation")
                    if bn: net_enc_h17 = batch_norm(net_enc_h17, is_training = net_is_training)
                with tf.variable_scope("MaxPool_18"):
                    net_enc_h18 = tf.nn.max_pool(net_enc_h17, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    net_enc_h18 = tf.reshape(net_enc_h18, [-1, 7*7*512])
                with tf.variable_scope('DenseLayer_19'):
                    net_enc_w19 = tf.Variable(tf.truncated_normal([7*7*512, 4096], stddev=0.001), name="Weights")
                    net_enc_b19 = tf.Variable(tf.constant(0., shape=[4096]), name="Biases")
                    net_enc_h19 = tf.nn.relu(tf.matmul(net_enc_h18, net_enc_w19) + net_enc_b19)
                    if bn: net_enc_h19 = batch_norm(net_enc_h19, is_training = net_is_training)
                with tf.variable_scope('DenseLayer_20'):
                    net_enc_w20 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.001), name="Weights")
                    net_enc_b20 = tf.Variable(tf.constant(0., shape=[4096]), name="Biases")
                    net_enc_h20 = tf.nn.relu(tf.matmul(net_enc_h19, net_enc_w20) + net_enc_b20)
                    if bn: net_enc_h20 = batch_norm(net_enc_h20, is_training = net_is_training)
                with tf.variable_scope('DenseLayer_21'):
                    net_enc_w21 = tf.Variable(tf.truncated_normal([4096, emb_size], stddev=0.001), name="Weights")
                    net_enc_b21 = tf.Variable(tf.constant(0., shape=[emb_size]), name="Biases")
                    net_enc_h21 = tf.matmul(net_enc_h20, net_enc_w21) + net_enc_b21
            with tf.variable_scope("Latent"):
                    net_enc_z = net_enc_h21
                    self._net_latent = tf.stack([net_enc_z, net_enc_z], axis=1)
            with tf.variable_scope("Decoder"):
                with tf.variable_scope("DenseLayer_1"):
                    net_dec_w1 = tf.Variable(tf.truncated_normal([emb_size,4096], stddev=0.001), name="Weights")
                    net_dec_b1 = tf.Variable(tf.constant(0., shape=[4096]), name="Biases")
                    net_dec_h1 = tf.nn.relu(tf.matmul(net_enc_z, net_dec_w1) + net_dec_b1)
                    if bn: net_dec_h1 = batch_norm(net_dec_h1, is_training = net_is_training)
                with tf.variable_scope("DenseLayer_2"):
                    net_dec_w2 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.001), name="Weights")
                    net_dec_b2 = tf.Variable(tf.constant(0., shape=[4096]), name="Biases")
                    net_dec_h2 = tf.nn.relu(tf.matmul(net_dec_h1, net_dec_w2) + net_dec_b2)
                    if bn: net_dec_h2 = batch_norm(net_dec_h2, is_training = net_is_training)
                with tf.variable_scope("DenseLayer_3"):
                    net_dec_w3 = tf.Variable(tf.truncated_normal([4096, 7*7*512], stddev=0.001), name="Weights")
                    net_dec_b3 = tf.Variable(tf.constant(0., shape=[7*7*512]), name="Biases")
                    net_dec_h3 = tf.nn.relu(tf.matmul(net_dec_h2, net_dec_w3) + net_dec_b3)
                    net_dec_h3 = tf.reshape(net_dec_h3, [batch_size, 7, 7, 512])
                    if bn: net_dec_h3 = batch_norm(net_dec_h3, is_training = net_is_training)
                with tf.variable_scope("DeconvLayer_4"):
                    net_dec_w4 = tf.Variable(tf.truncated_normal([3,3,512,256], stddev=0.001), name="Weights")
                    net_dec_b4 = tf.Variable(tf.constant(0., shape=[256]), name="Biases")
                    net_dec_s4 = [net_dec_h3.shape.as_list()[1]*2, net_dec_h3.shape.as_list()[2]*2]
                    net_dec_h4 = tf.image.resize_images(net_dec_h3, net_dec_s4, method=1)
                    net_dec_h4 = tf.nn.conv2d(net_dec_h4, net_dec_w4, strides=[1, 1, 1, 1], padding='SAME')
                    net_dec_h4 = tf.nn.relu(tf.add(net_dec_h4, net_dec_b4), name="Activation")
                    if bn: net_dec_h4 = batch_norm(net_dec_h4, is_training = net_is_training)
                with tf.variable_scope("DeconvLayer_5"):
                    net_dec_w5 = tf.Variable(tf.truncated_normal([3,3,256,128], stddev=0.001), name="Weights")
                    net_dec_b5 = tf.Variable(tf.constant(0., shape=[128]), name="Biases")
                    net_dec_s5 = [net_dec_h4.shape.as_list()[1]*2, net_dec_h4.shape.as_list()[2]*2]
                    net_dec_h5 = tf.image.resize_images(net_dec_h4, net_dec_s5, method=1)
                    net_dec_h5 = tf.nn.conv2d(net_dec_h5, net_dec_w5, strides=[1, 1, 1, 1], padding='SAME')
                    net_dec_h5 = tf.nn.relu(tf.add(net_dec_h5, net_dec_b5), name="Activation")
                    if bn: net_dec_h5 = batch_norm(net_dec_h5, is_training = net_is_training)
                with tf.variable_scope("DeconvLayer_6"):
                    net_dec_w6 = tf.Variable(tf.truncated_normal([3,3,128,64], stddev=0.001), name="Weights")
                    net_dec_b6 = tf.Variable(tf.constant(0., shape=[64]), name="Biases")
                    net_dec_s6 = [net_dec_h5.shape.as_list()[1]*2, net_dec_h5.shape.as_list()[2]*2]
                    net_dec_h6 = tf.image.resize_images(net_dec_h5, net_dec_s6, method=1)
                    net_dec_h6 = tf.nn.conv2d(net_dec_h6, net_dec_w6, strides=[1, 1, 1, 1], padding='SAME')
                    net_dec_h6 = tf.nn.relu(tf.add(net_dec_h6, net_dec_b6), name="Activation")
                    if bn: net_dec_h6 = batch_norm(net_dec_h6, is_training = net_is_training)
                with tf.variable_scope("DeconvLayer_7"):
                    net_dec_w7 = tf.Variable(tf.truncated_normal([3,3,64,3], stddev=0.001), name="Weights")
                    net_dec_b7 = tf.Variable(tf.constant(0., shape=[3]), name="Biases")
                    net_dec_s7 = [net_dec_h6.shape.as_list()[1]*2, net_dec_h6.shape.as_list()[2]*2]
                    net_dec_h7 = tf.image.resize_images(net_dec_h6, net_dec_s7, method=1)
                    net_dec_h7 = tf.nn.conv2d(net_dec_h7, net_dec_w7, strides=[1, 1, 1, 1], padding='SAME')
                    net_dec_h7 = tf.nn.relu(tf.add(net_dec_h7, net_dec_b7))
                    if bn: net_dec_h7 = batch_norm(net_dec_h7, is_training = net_is_training)
                with tf.variable_scope("DeconvLayer_8"):
                    net_dec_w8 = tf.Variable(tf.truncated_normal([3,3,3,3], stddev=0.001), name="Weights")
                    net_dec_b8 = tf.Variable(tf.constant(0., shape=[3]), name="Biases")
                    net_dec_s8 = [net_dec_h7.shape.as_list()[1]*2, net_dec_h7.shape.as_list()[2]*2]
                    net_dec_h8 = tf.image.resize_images(net_dec_h7, net_dec_s8, method=1)
                    net_dec_h8 = tf.nn.conv2d(net_dec_h8, net_dec_w8, strides=[1, 1, 1, 1], padding='SAME')
                    net_dec_h8 = tf.add(net_dec_h8, net_dec_b8)
                    net_dec_h8 = tf.reshape(net_dec_h8,[-1,signal_size*signal_size*3])
            with tf.variable_scope('Output'):
                self._net_output = tf.nn.sigmoid(net_dec_h8, name='Output')
                self._net_label = tf.placeholder(tf.float32, shape=[None, signal_size * signal_size * 3], name='Label')

            # Define Loss # ============================================================================================
            with tf.variable_scope('Loss'):
                with tf.variable_scope("ReconstructionLikelihood"):
                    net_loss_lkh = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(net_dec_h8, [batch_size, signal_size*signal_size*3]), labels=self._net_label)
                    net_loss_lkh = tf.reduce_sum(net_loss_lkh, reduction_indices=1)
                self._net_train_step = tf.constant([1.])
                net_annealing = tf.minimum(1.,0.01+self._net_train_step/ann_duration)
                self._net_loss = tf.reduce_mean(net_loss_lkh)

            # Define Accuracy # ========================================================================================
            with tf.variable_scope('Performance'):
                self._net_performance = self._net_loss

            # Define Optimizer # =======================================================================================
            with tf.variable_scope('Optimizer'):
                net_optimizer = tf.train.AdamOptimizer(lr)
                self._net_gradients = net_optimizer.compute_gradients(self._net_loss)
                self._net_optimize = net_optimizer.apply_gradients(self._net_gradients)


            # Define Train Dict # ======================================================================================
            self._net_train_dict = {net_is_training: True}
            self._net_test_dict = {net_is_training: False}

            # Define Summaries # =======================================================================================
            with tf.variable_scope('Summaries'):
                with tf.variable_scope("Scalars"):
                    tf.summary.scalar('Loss', self._net_loss)
                    tf.summary.scalar('LogLikelihood', tf.reduce_mean(-net_loss_lkh))
                    tf.summary.scalar('Performance', self._net_performance)

    def _load_weights(self, path):
        with h5py.File(path, mode='r') as f:
            with self._tf_graph.as_default():
                # ConvLayer1
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_1/Weights:0'),f['layer_1']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_1/Biases:0'),f['layer_1']['param_1'][:]))
                # ConvLayer2
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_2/Weights:0'),f['layer_3']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_2/Biases:0'),f['layer_3']['param_1'][:]))
                # ConvLayer4
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_4/Weights:0'),f['layer_6']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_4/Biases:0'),f['layer_6']['param_1'][:]))
                # ConvLayer5
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_5/Weights:0'),f['layer_8']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_5/Biases:0'),f['layer_8']['param_1'][:]))
                # ConvLayer7
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_7/Weights:0'), f['layer_11']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_7/Biases:0'), f['layer_11']['param_1'][:]))
                # ConvLayer8
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_8/Weights:0'), f['layer_13']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_8/Biases:0'), f['layer_13']['param_1'][:]))
                # ConvLayer9
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_9/Weights:0'), f['layer_15']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_9/Biases:0'), f['layer_15']['param_1'][:]))
                # ConvLayer11
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_11/Weights:0'), f['layer_18']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_11/Biases:0'), f['layer_18']['param_1'][:]))
                # ConvLayer12
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_12/Weights:0'), f['layer_20']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_12/Biases:0'), f['layer_20']['param_1'][:]))
                # ConvLayer13
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_13/Weights:0'), f['layer_22']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_13/Biases:0'), f['layer_22']['param_1'][:]))
                # ConvLayer15
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_15/Weights:0'), f['layer_25']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_15/Biases:0'), f['layer_25']['param_1'][:]))
                # ConvLayer16
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_16/Weights:0'), f['layer_27']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_16/Biases:0'), f['layer_27']['param_1'][:]))
                # ConvLayer17
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_17/Weights:0'), f['layer_29']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_17/Biases:0'), f['layer_29']['param_1'][:]))
                # DenseLayer19
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/DenseLayer_19/Weights:0'), f['layer_32']['param_0'][:]))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/DenseLayer_19/Biases:0'), f['layer_32']['param_1'][:]))
                # DenseLayer20
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/DenseLayer_20/Weights:0'), f['layer_34']['param_0'][:]))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/DenseLayer_20/Biases:0'), f['layer_34']['param_1'][:]))


class VGG16VAE(BaseNetwork):
    """
    A VAE based on VGG16 architecture and weights.
    """

    def _construct_arch(self, emb_size=10, signal_size=224, lr=1e-3, beta=1., ann_duration=1e4, bn=False):
        with self._tf_graph.as_default():
            # Define Network # =========================================================================================
            with tf.variable_scope('Input'):
                self._net_input = tf.placeholder(tf.float32, shape=[None, signal_size * signal_size * 3], name='Input')
                batch_size = tf.shape(self._net_input)[0]
                net_is_training = tf.placeholder(tf.bool, name="IsTraining")
                net_inp_input = tf.reshape(self._net_input, [batch_size, signal_size, signal_size, 3])
            with tf.variable_scope("Encoder"):
                with tf.variable_scope("ConvLayer_1"):
                    net_enc_w1 = tf.Variable(tf.truncated_normal([3,3,3,64], stddev=0.001), name="Weights", trainable=False)
                    net_enc_b1 = tf.Variable(tf.constant(0., shape=[64]), name="Biases", trainable=False)
                    net_enc_h1 = tf.nn.conv2d(net_inp_input ,net_enc_w1, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h1 = tf.nn.relu(tf.add(net_enc_h1, net_enc_b1), name="Activation")
                    if bn: net_enc_h1 = batch_norm(net_enc_h1, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_2"):
                    net_enc_w2 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.001), name="Weights", trainable=False)
                    net_enc_b2 = tf.Variable(tf.constant(0., shape=[64]), name="Biases", trainable=False)
                    net_enc_h2 = tf.nn.conv2d(net_enc_h1,net_enc_w2, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h2 = tf.nn.relu(tf.add(net_enc_h2, net_enc_b2), name="Activation")
                    if bn: net_enc_h2 = batch_norm(net_enc_h2, is_training = net_is_training)
                with tf.variable_scope("MaxPool_3"):
                    net_enc_h3 = tf.nn.max_pool(net_enc_h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("ConvLayer_4"):
                    net_enc_w4 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.001), name="Weights")
                    net_enc_b4 = tf.Variable(tf.constant(0., shape=[128]), name="Biases")
                    net_enc_h4 = tf.nn.conv2d(net_enc_h3,net_enc_w4, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h4 = tf.nn.relu(tf.add(net_enc_h4, net_enc_b4), name="Activation")
                    if bn: net_enc_h4 = batch_norm(net_enc_h4, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_5"):
                    net_enc_w5 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.001), name="Weights")
                    net_enc_b5 = tf.Variable(tf.constant(0., shape=[128]), name="Biases")
                    net_enc_h5 = tf.nn.conv2d(net_enc_h4,net_enc_w5, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h5 = tf.nn.relu(tf.add(net_enc_h5, net_enc_b5), name="Activation")
                    if bn: net_enc_h5 = batch_norm(net_enc_h5, is_training = net_is_training)
                with tf.variable_scope("MaxPool_6"):
                    net_enc_h6 = tf.nn.max_pool(net_enc_h5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("ConvLayer_7"):
                    net_enc_w7 = tf.Variable(tf.truncated_normal([3,3,128, 256], stddev=0.001), name="Weights")
                    net_enc_b7 = tf.Variable(tf.constant(0., shape=[256]), name="Biases")
                    net_enc_h7 = tf.nn.conv2d(net_enc_h6,net_enc_w7, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h7 = tf.nn.relu(tf.add(net_enc_h7, net_enc_b7), name="Activation")
                    if bn: net_enc_h7 = batch_norm(net_enc_h7, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_8"):
                    net_enc_w8 = tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.001), name="Weights")
                    net_enc_b8 = tf.Variable(tf.constant(0., shape=[256]), name="Biases")
                    net_enc_h8 = tf.nn.conv2d(net_enc_h7, net_enc_w8, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h8 = tf.nn.relu(tf.add(net_enc_h8, net_enc_b8), name="Activation")
                    if bn: net_enc_h8 = batch_norm(net_enc_h8, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_9"):
                    net_enc_w9 = tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.001), name="Weights")
                    net_enc_b9 = tf.Variable(tf.constant(0., shape=[256]), name="Biases")
                    net_enc_h9 = tf.nn.conv2d(net_enc_h8,net_enc_w9, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h9 = tf.nn.relu(tf.add(net_enc_h9, net_enc_b9), name="Activation")
                    if bn: net_enc_h9 = batch_norm(net_enc_h9, is_training = net_is_training)
                with tf.variable_scope("MaxPool_10"):
                    net_enc_h10 = tf.nn.max_pool(net_enc_h9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("ConvLayer_11"):
                    net_enc_w11 = tf.Variable(tf.truncated_normal([3,3,256,512], stddev=0.001), name="Weights")
                    net_enc_b11 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h11 = tf.nn.conv2d(net_enc_h10,net_enc_w11, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h11 = tf.nn.relu(tf.add(net_enc_h11, net_enc_b11), name="Activation")
                    if bn: net_enc_h11 = batch_norm(net_enc_h11, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_12"):
                    net_enc_w12 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001), name="Weights")
                    net_enc_b12 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h12 = tf.nn.conv2d(net_enc_h11, net_enc_w12, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h12 = tf.nn.relu(tf.add(net_enc_h12, net_enc_b12), name="Activation")
                    if bn: net_enc_h12 = batch_norm(net_enc_h12, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_13"):
                    net_enc_w13 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001), name="Weights")
                    net_enc_b13 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h13 = tf.nn.conv2d(net_enc_h12,net_enc_w13, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h13 = tf.nn.relu(tf.add(net_enc_h13, net_enc_b13), name="Activation")
                    if bn: net_enc_h13 = batch_norm(net_enc_h13, is_training = net_is_training)
                with tf.variable_scope("MaxPool_14"):
                    net_enc_h14 = tf.nn.max_pool(net_enc_h13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("ConvLayer_15"):
                    net_enc_w15 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001), name="Weights")
                    net_enc_b15 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h15 = tf.nn.conv2d(net_enc_h14,net_enc_w15, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h15 = tf.nn.relu(tf.add(net_enc_h15, net_enc_b15), name="Activation")
                    if bn: net_enc_h15 = batch_norm(net_enc_h15, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_16"):
                    net_enc_w16 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001), name="Weights")
                    net_enc_b16 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h16 = tf.nn.conv2d(net_enc_h15, net_enc_w16, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h16 = tf.nn.relu(tf.add(net_enc_h16, net_enc_b16), name="Activation")
                    if bn: net_enc_h16 = batch_norm(net_enc_h16, is_training = net_is_training)
                with tf.variable_scope("ConvLayer_17"):
                    net_enc_w17 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001), name="Weights")
                    net_enc_b17 = tf.Variable(tf.constant(0., shape=[512]), name="Biases")
                    net_enc_h17 = tf.nn.conv2d(net_enc_h16,net_enc_w17, strides=[1, 1, 1, 1], padding='SAME')
                    net_enc_h17 = tf.nn.relu(tf.add(net_enc_h17, net_enc_b17), name="Activation")
                    if bn: net_enc_h17 = batch_norm(net_enc_h17, is_training = net_is_training)
                with tf.variable_scope("MaxPool_18"):
                    net_enc_h18 = tf.nn.max_pool(net_enc_h17, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    net_enc_h18 = tf.reshape(net_enc_h18, [-1, 7*7*512])
                with tf.variable_scope('OutputLayer_z0_mu'):
                    net_enc_w21 = tf.Variable(tf.truncated_normal([7*7*512, emb_size], stddev=0.001), name="Weights")
                    net_enc_b21 = tf.Variable(tf.constant(0., shape=[emb_size]), name="Biases")
                    net_enc_mu = tf.matmul(net_enc_h18, net_enc_w21) + net_enc_b21
                    if bn: net_enc_mu = batch_norm(net_enc_mu, is_training = net_is_training)
                with tf.variable_scope('OutputLayer_z0_log_sigma'):
                    net_enc_w22 = tf.Variable(tf.truncated_normal([7*7*512, emb_size], stddev=0.001), name="Weights")
                    net_enc_b22 = tf.Variable(tf.constant(0., shape=[emb_size]), name="Biases")
                    net_enc_log_var = tf.matmul(net_enc_h18, net_enc_w22) + net_enc_b22
                    if bn: net_enc_log_var = batch_norm(net_enc_log_var, is_training = net_is_training)
            with tf.variable_scope("Latent"):
                with tf.variable_scope("ReparametrizationLayer"):
                    net_rep_eps = tf.random_normal([emb_size])
                    net_rep_mu = net_enc_mu
                    net_rep_sigma = tf.exp(net_enc_log_var/2.)
                    self._net_latent = tf.stack([net_rep_mu, net_rep_sigma], axis=1)
                    net_rep_z = net_enc_mu + tf.multiply(net_rep_eps, net_rep_sigma)
            with tf.variable_scope("Decoder"):
                with tf.variable_scope("DenseLayer_1"):
                    net_dec_w1 = tf.Variable(tf.truncated_normal([emb_size,7*7*512], stddev=0.001), name="Weights")
                    net_dec_b1 = tf.Variable(tf.constant(0., shape=[7*7*512]), name="Biases")
                    net_dec_h1 = tf.matmul(net_rep_z, net_dec_w1) + net_dec_b1
                    net_dec_h1 = tf.reshape(net_dec_h1, [batch_size, 7, 7, 512])
                    if bn: net_dec_h1 = batch_norm(net_dec_h1, is_training = net_is_training)
                with tf.variable_scope("DeconvLayer_4"):
                    net_dec_w4 = tf.Variable(tf.truncated_normal([3,3,512,256], stddev=0.001), name="Weights")
                    net_dec_b4 = tf.Variable(tf.constant(0., shape=[256]), name="Biases")
                    net_dec_s4 = [net_dec_h1.shape.as_list()[1]*2, net_dec_h1.shape.as_list()[2]*2]
                    net_dec_h4 = tf.image.resize_images(net_dec_h1, net_dec_s4, method=1)
                    net_dec_h4 = tf.nn.conv2d(net_dec_h4, net_dec_w4, strides=[1, 1, 1, 1], padding='SAME')
                    net_dec_h4 = tf.add(net_dec_h4, net_dec_b4)
                    if bn: net_dec_h4 = batch_norm(net_dec_h4, is_training = net_is_training)
                with tf.variable_scope("DeconvLayer_5"):
                    net_dec_w5 = tf.Variable(tf.truncated_normal([3,3,256,128], stddev=0.001), name="Weights")
                    net_dec_b5 = tf.Variable(tf.constant(0., shape=[128]), name="Biases")
                    net_dec_s5 = [net_dec_h4.shape.as_list()[1]*2, net_dec_h4.shape.as_list()[2]*2]
                    net_dec_h5 = tf.image.resize_images(net_dec_h4, net_dec_s5, method=1)
                    net_dec_h5 = tf.nn.conv2d(net_dec_h5, net_dec_w5, strides=[1, 1, 1, 1], padding='SAME')
                    net_dec_h5 = tf.add(net_dec_h5, net_dec_b5)
                    if bn: net_dec_h5 = batch_norm(net_dec_h5, is_training = net_is_training)
                with tf.variable_scope("DeconvLayer_6"):
                    net_dec_w6 = tf.Variable(tf.truncated_normal([3,3,128,64], stddev=0.001), name="Weights")
                    net_dec_b6 = tf.Variable(tf.constant(0., shape=[64]), name="Biases")
                    net_dec_s6 = [net_dec_h5.shape.as_list()[1]*2, net_dec_h5.shape.as_list()[2]*2]
                    net_dec_h6 = tf.image.resize_images(net_dec_h5, net_dec_s6, method=1)
                    net_dec_h6 = tf.nn.conv2d(net_dec_h6, net_dec_w6, strides=[1, 1, 1, 1], padding='SAME')
                    net_dec_h6 = tf.add(net_dec_h6, net_dec_b6)
                    if bn: net_dec_h6 = batch_norm(net_dec_h6, is_training = net_is_training)
                with tf.variable_scope("DeconvLayer_7"):
                    net_dec_w7 = tf.Variable(tf.truncated_normal([3,3,64,3], stddev=0.001), name="Weights")
                    net_dec_b7 = tf.Variable(tf.constant(0., shape=[3]), name="Biases")
                    net_dec_s7 = [net_dec_h6.shape.as_list()[1]*2, net_dec_h6.shape.as_list()[2]*2]
                    net_dec_h7 = tf.image.resize_images(net_dec_h6, net_dec_s7, method=1)
                    net_dec_h7 = tf.nn.conv2d(net_dec_h7, net_dec_w7, strides=[1, 1, 1, 1], padding='SAME')
                    net_dec_h7 = tf.add(net_dec_h7, net_dec_b7)
                    if bn: net_dec_h7 = batch_norm(net_dec_h7, is_training = net_is_training)
                with tf.variable_scope("DeconvLayer_8"):
                    net_dec_w8 = tf.Variable(tf.truncated_normal([3,3,3,3], stddev=0.001), name="Weights")
                    net_dec_b8 = tf.Variable(tf.constant(0., shape=[3]), name="Biases")
                    net_dec_s8 = [net_dec_h7.shape.as_list()[1]*2, net_dec_h7.shape.as_list()[2]*2]
                    net_dec_h8 = tf.image.resize_images(net_dec_h7, net_dec_s8, method=1)
                    net_dec_h8 = tf.nn.conv2d(net_dec_h8, net_dec_w8, strides=[1, 1, 1, 1], padding='SAME')
                    net_dec_h8 = tf.add(net_dec_h8, net_dec_b8)
                    net_dec_h8 = tf.reshape(net_dec_h8,[-1,signal_size*signal_size*3])
            with tf.variable_scope('Output'):
                self._net_output = net_dec_h8
                self._net_label = tf.placeholder(tf.float32, shape=[None, signal_size * signal_size * 3], name='Label')

            # Define Loss # ============================================================================================
            with tf.variable_scope('Loss'):
                with tf.variable_scope("LatentDivergence"):
                    net_loss_kld = 1 + net_enc_log_var - tf.square(net_enc_mu) - tf.exp(net_enc_log_var)
                    net_loss_kld = - beta * 0.5 * tf.reduce_sum(net_loss_kld, reduction_indices=1)
                with tf.variable_scope("ReconstructionLikelihood"):
                    net_loss_lkh = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(net_dec_h8, [batch_size, signal_size*signal_size*3]), labels=self._net_label)
                    net_loss_lkh = tf.reduce_sum(net_loss_lkh, reduction_indices=1)
                self._net_train_step = tf.constant([1.])
                net_annealing = tf.minimum(1.,0.01+self._net_train_step/ann_duration)
                self._net_loss = tf.reduce_mean(net_annealing*(net_loss_kld) + net_loss_lkh)

            # Define Accuracy # ========================================================================================
            with tf.variable_scope('Performance'):
                self._net_performance = self._net_loss

            # Define Optimizer # =======================================================================================
            with tf.variable_scope('Optimizer'):
                net_optimizer = tf.train.AdamOptimizer(lr)
                self._net_gradients = net_optimizer.compute_gradients(self._net_loss)
                self._net_optimize = net_optimizer.apply_gradients(self._net_gradients)


            # Define Train Dict # ======================================================================================
            self._net_train_dict = {net_is_training: True}
            self._net_test_dict = {net_is_training: False}

            # Define Summaries # =======================================================================================
            with tf.variable_scope('Summaries'):
                with tf.variable_scope("Scalars"):
                    tf.summary.scalar('Loss', self._net_loss)
                    tf.summary.scalar('LogLikelihood', tf.reduce_mean(-net_loss_lkh))
                    tf.summary.scalar('KL-Div', tf.reduce_mean(net_loss_kld))
                    tf.summary.scalar('Performance', self._net_performance)
                    tf.summary.scalar('Annealing', net_annealing[0])

    def _load_weights(self, path):
        with h5py.File(path, mode='r') as f:
            with self._tf_graph.as_default():
                # ConvLayer1
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_1/Weights:0'),f['layer_1']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_1/Biases:0'),f['layer_1']['param_1'][:]))
                # ConvLayer2
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_2/Weights:0'),f['layer_3']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_2/Biases:0'),f['layer_3']['param_1'][:]))
                # ConvLayer4
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_4/Weights:0'),f['layer_6']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_4/Biases:0'),f['layer_6']['param_1'][:]))
                # ConvLayer5
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_5/Weights:0'),f['layer_8']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_5/Biases:0'),f['layer_8']['param_1'][:]))
                # ConvLayer7
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_7/Weights:0'), f['layer_11']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_7/Biases:0'), f['layer_11']['param_1'][:]))
                # ConvLayer8
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_8/Weights:0'), f['layer_13']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_8/Biases:0'), f['layer_13']['param_1'][:]))
                # ConvLayer9
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_9/Weights:0'), f['layer_15']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_9/Biases:0'), f['layer_15']['param_1'][:]))
                # ConvLayer11
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_11/Weights:0'), f['layer_18']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_11/Biases:0'), f['layer_18']['param_1'][:]))
                # ConvLayer12
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_12/Weights:0'), f['layer_20']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_12/Biases:0'), f['layer_20']['param_1'][:]))
                # ConvLayer13
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_13/Weights:0'), f['layer_22']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_13/Biases:0'), f['layer_22']['param_1'][:]))
                # ConvLayer15
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_15/Weights:0'), f['layer_25']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_15/Biases:0'), f['layer_25']['param_1'][:]))
                # ConvLayer16
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_16/Weights:0'), f['layer_27']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_16/Biases:0'), f['layer_27']['param_1'][:]))
                # ConvLayer17
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_17/Weights:0'), f['layer_29']['param_0'][:].T))
                self._tf_session.run(tf.assign(self.get_tensor('Encoder/ConvLayer_17/Biases:0'), f['layer_29']['param_1'][:]))


class DenseRadialVAE(BaseNetwork):
    """
    A Simple Dense VAE following specifications from Irina Higgins's paper, using Radial Flow
    """

    def _construct_arch(self, emb_size=10, signal_size=70**2, nb_trans=10, beta=1., lr=1e-3, ann_duration=1e4):
        with self._tf_graph.as_default():
            # Define Network # =========================================================================================
            with tf.variable_scope('Input'):
                self._net_input = tf.placeholder(tf.float32, shape=[None, 70 * 70], name='Input')
            with tf.variable_scope("Encoder"):
                with tf.variable_scope("ReluLayer_1"):
                    net_enc_w1 = tf.Variable(tf.truncated_normal([signal_size, 1200], stddev=0.001), name="Weights")
                    net_enc_b1 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_enc_h1 = tf.nn.relu(tf.matmul(self._net_input, net_enc_w1) + net_enc_b1)
                with tf.variable_scope("ReluLayer_2"):
                    net_enc_w2 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_enc_b2 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_enc_h2 = tf.nn.relu(tf.matmul(net_enc_h1, net_enc_w2) + net_enc_b2)
                with tf.variable_scope('OutputLayer_z0_mu'):
                    net_enc_w3 = tf.Variable(tf.truncated_normal([1200, emb_size], stddev=0.001), name="Weights")
                    net_enc_b3 = tf.Variable(tf.constant(0., shape=[emb_size]), name="Biases")
                    net_enc_mu = tf.matmul(net_enc_h2, net_enc_w3) + net_enc_b3
                with tf.variable_scope('OutputLayer_z0_log_sigma'):
                    net_enc_w4 = tf.Variable(tf.truncated_normal([1200, emb_size], stddev=0.001), name="Weights")
                    net_enc_b4 = tf.Variable(tf.constant(0., shape=[emb_size]), name="Biases")
                    net_enc_log_var = tf.matmul(net_enc_h2, net_enc_w4) + net_enc_b4
                with tf.variable_scope("OutputLayer_pf_c"):
                    net_enc_w5 = tf.Variable(tf.truncated_normal([1200, nb_trans * emb_size], stddev=0.001), name="Weights")
                    net_enc_b5 = tf.Variable(tf.constant(0., shape=[nb_trans * emb_size]), name="Biases")
                    net_enc_cs = tf.matmul(net_enc_h2, net_enc_w5) + net_enc_b5
                    net_enc_cs = tf.reshape(net_enc_cs, [-1, nb_trans, emb_size])
                with tf.variable_scope('OutputLayer_pf_b'):
                    net_enc_w6 = tf.Variable(tf.truncated_normal([1200, nb_trans], stddev=0.001), name="Weights")
                    net_enc_b6 = tf.Variable(tf.constant(0., shape=[nb_trans]), name="Biases")
                    net_enc_bs = tf.matmul(net_enc_h2, net_enc_w6) + net_enc_b6
                with tf.variable_scope('OutputLayer_pf_a'):
                    net_enc_w7 = tf.Variable(tf.truncated_normal([1200, nb_trans], stddev=0.001), name="Weights")
                    net_enc_b7 = tf.Variable(tf.constant(0., shape=[nb_trans]), name="Biases")
                    net_enc_as = tf.matmul(net_enc_h2, net_enc_w7) + net_enc_b7
            with tf.variable_scope("Latent"):
                with tf.variable_scope("ReparametrizationLayer"):
                    net_rep_eps = tf.random_normal([emb_size])
                    net_rep_mu = net_enc_mu
                    net_rep_sigma = tf.exp(net_enc_log_var/2.)
                    self._net_latent = tf.stack([net_rep_mu, net_rep_sigma], axis=1)
                    net_rep_z = net_enc_mu + tf.multiply(net_rep_eps, net_rep_sigma)
                with tf.variable_scope("RadialFlow"):
                    net_flow_log_det_jacs = 0
                    net_flow_output = net_rep_z
                    for k in range(nb_trans):
                        with tf.variable_scope("RadialFlowLayer_%i"%k):
                            c = net_enc_cs[:, k]
                            a = tf.abs(net_enc_as[:, k])
                            b = net_enc_bs[:, k]
                            # We enforce the invertibility (Annex A2)
                            b_c = -a + tf.nn.softplus(b)
                            # We compute flow (Eq 14)
                            r = tf.norm(net_flow_output - c, axis=1)
                            h = 1 / (a + r)
                            bh = tf.multiply(b_c, h)
                            flow = tf.einsum('i,ij->ij', bh, net_flow_output - c)
                            net_flow_output = net_flow_output + flow
                            # We compute log det (Eq 14)
                            log_det_jac = - 1 / tf.square(a + r)
                            log_det_jac = log_det_jac * tf.expand_dims(b_c, axis=1) * r
                            log_det_jac = log_det_jac + 1 + bh
                            log_det_jac = log_det_jac * tf.pow(1 + bh, emb_size - 1)
                            log_det_jac = tf.log(tf.clip_by_value(log_det_jac, 1e-4, 1e6))
                            net_flow_log_det_jacs = net_flow_log_det_jacs + log_det_jac
                        self._net_latent = tf.stack([net_rep_z, net_flow_output], axis=1)
            with tf.variable_scope("Decoder"):
                with tf.variable_scope("ReluLayer_1"):
                    net_dec_w1 = tf.Variable(tf.truncated_normal([emb_size, 1200], stddev=0.001), name="Weights")
                    net_dec_b1 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_h1 = tf.nn.relu(tf.matmul(net_flow_output, net_dec_w1) + net_dec_b1)
                with tf.variable_scope("ReluLayer_2"):
                    net_dec_w2 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_dec_b2 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_h2 = tf.nn.relu(tf.matmul(net_dec_h1, net_dec_w2) + net_dec_b2)
                """
                with tf.variable_scope("ReluLayer_3"):
                    net_dec_w3 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_dec_b3 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_h3 = tf.nn.tanh(tf.matmul(net_dec_h2, net_dec_w3) + net_dec_b3)
                """
                with tf.variable_scope("ReluLayer_4"):
                    net_dec_w4 = tf.Variable(tf.truncated_normal([1200, signal_size], stddev=0.001), name="Weights")
                    net_dec_b4 = tf.Variable(tf.constant(0., shape=[signal_size]), name="Baises")
                    net_dec_h4 = tf.matmul(net_dec_h2, net_dec_w4) + net_dec_b4
            with tf.variable_scope('Output'):
                self._net_output = tf.nn.sigmoid(net_dec_h4, name='Output')
                self._net_label = tf.placeholder(tf.float32, shape=[None, signal_size], name='Label')

            # Define Loss # ============================================================================================
            with tf.variable_scope('Loss'):
                with tf.variable_scope("LatentDivergence"):
                    net_loss_kld = 1 + net_enc_log_var - tf.square(net_enc_mu) - tf.exp(net_enc_log_var)
                    net_loss_kld = - beta * 0.5 * tf.reduce_sum(net_loss_kld, reduction_indices=1)
                with tf.variable_scope("ReconstructionLikelihood"):
                    net_loss_lkh = tf.nn.sigmoid_cross_entropy_with_logits(logits=net_dec_h4, labels=self._net_input)
                    net_loss_lkh = tf.reduce_sum(net_loss_lkh, reduction_indices=1)
                with tf.variable_scope("FlowLogDetJacs"):
                    net_loss_ldj = - net_flow_log_det_jacs
                self._net_train_step = tf.constant([1.])
                net_annealing = tf.minimum(1.,0.01+self._net_train_step/ann_duration)
                self._net_loss = tf.reduce_mean(net_annealing*(net_loss_kld + net_loss_lkh) + net_loss_ldj)

            # Define Accuracy # ========================================================================================
            with tf.variable_scope('Performance'):
                self._net_performance = self._net_loss

            # Define Optimizer # =======================================================================================
            with tf.variable_scope('Optimizer'):
                net_optimizer = tf.train.AdamOptimizer(lr)
                self._net_gradients = net_optimizer.compute_gradients(self._net_loss)
                self._net_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self._net_gradients]
                self._net_optimize = net_optimizer.apply_gradients(self._net_gradients)

            # Define Train Dict # ======================================================================================
            self._net_train_dict = {}
            self._net_test_dict = {}

            # Define Summaries # =======================================================================================
            with tf.variable_scope('Summaries'):
                with tf.variable_scope("Scalars"):
                    tf.summary.scalar('Loss', self._net_loss)
                    tf.summary.scalar('Kullback-Liebler', tf.reduce_mean(net_loss_kld))
                    tf.summary.scalar('LogLikelihood', tf.reduce_mean(-net_loss_lkh))
                    tf.summary.scalar('LogDetJacs', tf.reduce_mean(-net_loss_ldj))
                    tf.summary.scalar('Performance', self._net_performance)
                    tf.summary.scalar('Annealing', net_annealing[0])


class DensePlanarVAE(BaseNetwork):
    """
    A Simple Dense VAE following specifications from Irina Higgins's paper, using Planar Flow
    """

    def _construct_arch(self, emb_size=10, signal_size=70**2, nb_trans=10, beta=1., lr=1e-3, ann_duration=1e4):
        with self._tf_graph.as_default():
            # Define Network # =========================================================================================
            with tf.variable_scope('Input'):
                self._net_input = tf.placeholder(tf.float32, shape=[None, 70 * 70], name='Input')
            with tf.variable_scope("Encoder"):
                with tf.variable_scope("ReluLayer_1"):
                    net_enc_w1 = tf.Variable(tf.truncated_normal([signal_size, 1200], stddev=0.001), name="Weights")
                    net_enc_b1 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_enc_h1 = tf.nn.relu(tf.matmul(self._net_input, net_enc_w1) + net_enc_b1)
                with tf.variable_scope("ReluLayer_2"):
                    net_enc_w2 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_enc_b2 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_enc_h2 = tf.nn.relu(tf.matmul(net_enc_h1, net_enc_w2) + net_enc_b2)
                with tf.variable_scope('OutputLayer_z0_mu'):
                    net_enc_w3 = tf.Variable(tf.truncated_normal([1200, emb_size], stddev=0.001), name="Weights")
                    net_enc_b3 = tf.Variable(tf.constant(0., shape=[emb_size]), name="Biases")
                    net_enc_mu = tf.matmul(net_enc_h2, net_enc_w3) + net_enc_b3
                with tf.variable_scope('OutputLayer_z0_log_sigma'):
                    net_enc_w4 = tf.Variable(tf.truncated_normal([1200, emb_size], stddev=0.001), name="Weights")
                    net_enc_b4 = tf.Variable(tf.constant(0., shape=[emb_size]), name="Biases")
                    net_enc_log_var = tf.matmul(net_enc_h2, net_enc_w4) + net_enc_b4
                with tf.variable_scope("OutputLayer_pf_w"):
                    net_enc_w5 = tf.Variable(tf.truncated_normal([1200, nb_trans * emb_size], stddev=0.001), name="Weights")
                    net_enc_b5 = tf.Variable(tf.constant(0., shape=[nb_trans * emb_size]), name="Biases")
                    net_enc_ws = tf.matmul(net_enc_h2, net_enc_w5) + net_enc_b5
                    net_enc_ws = tf.reshape(net_enc_ws, [-1, nb_trans, emb_size])
                with tf.variable_scope('OutputLayer_pf_b'):
                    net_enc_w6 = tf.Variable(tf.truncated_normal([1200, nb_trans], stddev=0.001), name="Weights")
                    net_enc_b6 = tf.Variable(tf.constant(0., shape=[nb_trans]), name="Biases")
                    net_enc_bs = tf.matmul(net_enc_h2, net_enc_w6) + net_enc_b6
                with tf.variable_scope('OutputLayer_pf_u'):
                    net_enc_w7 = tf.Variable(tf.truncated_normal([1200, nb_trans * emb_size], stddev=0.001), name="Weights")
                    net_enc_b7 = tf.Variable(tf.constant(0., shape=[nb_trans * emb_size]), name="Biases")
                    net_enc_us = tf.matmul(net_enc_h2, net_enc_w7) + net_enc_b7
                    net_enc_us = tf.reshape(net_enc_us, [-1, nb_trans, emb_size])
            with tf.variable_scope("Latent"):
                with tf.variable_scope("ReparametrizationLayer"):
                    net_rep_eps = tf.random_normal([emb_size])
                    net_rep_mu = net_enc_mu
                    net_rep_sigma = tf.exp(net_enc_log_var/2.)
                    self._net_latent = tf.stack([net_rep_mu, net_rep_sigma], axis=1)
                    net_rep_z = net_enc_mu + tf.multiply(net_rep_eps, net_rep_sigma)
                with tf.variable_scope("PlanarFlow"):
                    net_flow_log_det_jacs = 0
                    net_flow_output = net_rep_z
                    for k in range(nb_trans):
                        with tf.variable_scope("PlanarFlowLayer_%i"%k):
                            u = net_enc_us[:, k]
                            w = net_enc_ws[:, k]
                            b = net_enc_bs[:, k]
                            # We enforce the invertibility (Annex A1)
                            u_c = tf.einsum('ij,ij->i', w, u)
                            u_c = -1 + tf.log(tf.clip_by_value(1 + tf.exp(u_c), 1e-4, 1e6)) - u_c
                            u_c = tf.expand_dims(u_c, 1) * w / tf.norm(w, axis=1, keep_dims=True)
                            u_c = u_c + u
                            # We compute flow (Equation 10)
                            flow = tf.tanh(tf.einsum('ij,ij->i', net_flow_output, w) + b)
                            net_flow_output = net_flow_output + u_c * tf.expand_dims(flow, 1)
                            # We compute log det (Equation 11, 12)
                            log_det_jac = tf.einsum('ij,ij->i', net_flow_output, w) + b
                            log_det_jac = 1.0 - tf.square(tf.tanh(log_det_jac))
                            log_det_jac = tf.expand_dims(log_det_jac, 1) * w
                            log_det_jac = tf.einsum('ij,ij->i', log_det_jac, u_c)
                            log_det_jac = tf.log(tf.clip_by_value(tf.abs(1 + log_det_jac), 1e-4, 1e6))
                            net_flow_log_det_jacs = net_flow_log_det_jacs + log_det_jac
                        self._net_latent = tf.stack([net_rep_z, net_flow_output], axis=1)
            with tf.variable_scope("Decoder"):
                with tf.variable_scope("ReluLayer_1"):
                    net_dec_w1 = tf.Variable(tf.truncated_normal([emb_size, 1200], stddev=0.001), name="Weights")
                    net_dec_b1 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_h1 = tf.nn.relu(tf.matmul(net_flow_output, net_dec_w1) + net_dec_b1)
                with tf.variable_scope("ReluLayer_2"):
                    net_dec_w2 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_dec_b2 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_h2 = tf.nn.relu(tf.matmul(net_dec_h1, net_dec_w2) + net_dec_b2)
                """
                with tf.variable_scope("ReluLayer_3"):
                    net_dec_w3 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_dec_b3 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_h3 = tf.nn.relu(tf.matmul(net_dec_h2, net_dec_w3) + net_dec_b3)
                """
                with tf.variable_scope("ReluLayer_4"):
                    net_dec_w4 = tf.Variable(tf.truncated_normal([1200, signal_size], stddev=0.001), name="Weights")
                    net_dec_b4 = tf.Variable(tf.constant(0., shape=[signal_size]), name="Baises")
                    net_dec_h4 = tf.matmul(net_dec_h2, net_dec_w4) + net_dec_b4
            with tf.variable_scope('Output'):
                self._net_output = tf.nn.sigmoid(net_dec_h4, name='Output')
                self._net_label = tf.placeholder(tf.float32, shape=[None, signal_size], name='Label')

            # Define Loss # ============================================================================================
            with tf.variable_scope('Loss'):
                with tf.variable_scope("LatentDivergence"):
                    net_loss_kld = 1 + net_enc_log_var - tf.square(net_enc_mu) - tf.exp(net_enc_log_var)
                    net_loss_kld = - beta * 0.5 * tf.reduce_sum(net_loss_kld, reduction_indices=1)
                with tf.variable_scope("ReconstructionLikelihood"):
                    net_loss_lkh = tf.nn.sigmoid_cross_entropy_with_logits(logits=net_dec_h4, labels=self._net_input)
                    net_loss_lkh = tf.reduce_sum(net_loss_lkh, reduction_indices=1)
                with tf.variable_scope("FlowLogDetJacs"):
                    net_loss_ldj = - net_flow_log_det_jacs
                self._net_train_step = tf.constant([1.])
                net_annealing = tf.minimum(1.,0.01+self._net_train_step/ann_duration)
                self._net_loss = tf.reduce_mean(net_annealing*(net_loss_kld + net_loss_lkh) + net_loss_ldj)

            # Define Accuracy # ========================================================================================
            with tf.variable_scope('Performance'):
                self._net_performance = self._net_loss

            # Define Optimizer # =======================================================================================
            with tf.variable_scope('Optimizer'):
                net_optimizer = tf.train.AdamOptimizer(lr)
                self._net_gradients = net_optimizer.compute_gradients(self._net_loss)
                self._net_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self._net_gradients]
                self._net_optimize = net_optimizer.apply_gradients(self._net_gradients)

            # Define Train Dict # ======================================================================================
            self._net_train_dict = {}
            self._net_test_dict = {}

            # Define Summaries # =======================================================================================
            with tf.variable_scope('Summaries'):
                with tf.variable_scope("Scalars"):
                    tf.summary.scalar('Loss', self._net_loss)
                    tf.summary.scalar('Kullback-Liebler', tf.reduce_mean(net_loss_kld))
                    tf.summary.scalar('LogLikelihood', tf.reduce_mean(-net_loss_lkh))
                    tf.summary.scalar('LogDetJacs', tf.reduce_mean(-net_loss_ldj))
                    tf.summary.scalar('Performance', self._net_performance)
                    tf.summary.scalar('Annealing', net_annealing[0])


class DenseVAE(BaseNetwork):
    """
    A Simple Dense VAE following specifications from Irina Higgins's paper
    """

    def _construct_arch(self, lr=1e-4, beta=1., emb_size=10, signal_size=70):

        with self._tf_graph.as_default():

            # Define Network # =========================================================================================

            with tf.variable_scope('Input'):
                self._net_input = tf.placeholder(tf.float32, shape=[None, signal_size**2], name='Input')
                net_is_training = tf.placeholder(tf.bool, name="IsTraining")
            with tf.variable_scope("Encoder"):
                with tf.variable_scope("DenseLayer_1"):
                    net_enc_w1 = tf.Variable(tf.truncated_normal([signal_size**2, 1200], stddev=0.001), name="Weights")
                    net_enc_b1 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_enc_h1 = tf.nn.relu(tf.matmul(self._net_input, net_enc_w1) + net_enc_b1)
                with tf.variable_scope("DenseLayer_2"):
                    net_enc_w2 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_enc_b2 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_enc_h2 = tf.nn.relu(tf.matmul(net_enc_h1, net_enc_w2) + net_enc_b2)
                with tf.variable_scope("Output_layer_mu"):
                    net_enc_w3 = tf.Variable(tf.truncated_normal([1200, emb_size], stddev=0.001), name="Weights")
                    net_enc_b3 = tf.Variable(tf.constant(0., shape=[emb_size]), name="Biases")
                    net_enc_mu = tf.matmul(net_enc_h2, net_enc_w3) + net_enc_b3
                with tf.variable_scope("OutputLayer_logvar"):
                    net_enc_w4 = tf.Variable(tf.truncated_normal([1200, emb_size], stddev=0.001), name="Weights")
                    net_enc_b4 = tf.Variable(tf.constant(0., shape=[emb_size]), name="Biases")
                    net_enc_log_var = tf.matmul(net_enc_h2, net_enc_w4) + net_enc_b4
            with tf.variable_scope("Reparametrization"):
                net_rep_eps = tf.random_normal([emb_size])
                net_rep_mu = net_enc_mu
                net_rep_sigma = tf.exp(net_enc_log_var/2.)
                self._net_latent = tf.stack([net_rep_mu, net_rep_sigma], axis=1)
                net_rep_z = net_enc_mu + tf.multiply(net_rep_eps, net_rep_sigma)
            with tf.variable_scope("Decoder"):
                with tf.variable_scope("DenseLayer_1"):
                    net_dec_w1 = tf.Variable(tf.truncated_normal([emb_size, 1200], stddev=0.001), name="Weights")
                    net_dec_b1 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_h1 = tf.nn.relu(tf.matmul(net_rep_z, net_dec_w1) + net_dec_b1)
                with tf.variable_scope("DenseLayer_2"):
                    net_dec_w2 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_dec_b2 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_h2 = tf.nn.relu(tf.matmul(net_dec_h1, net_dec_w2) + net_dec_b2)
                with tf.variable_scope("Output_layer_4"):
                    net_dec_w4 = tf.Variable(tf.truncated_normal([1200, signal_size**2], stddev=0.001), name="Weights")
                    net_dec_b4 = tf.Variable(tf.constant(0., shape=[signal_size**2]), name="Biases")
                    net_dec_h4 = tf.matmul(net_dec_h2, net_dec_w4) + net_dec_b4
            with tf.variable_scope('Output'):
                self._net_output = tf.nn.sigmoid(net_dec_h4, name='Output')
                self._net_label = tf.placeholder(tf.float32, shape=[None, signal_size**2], name='Label')

            # Define Loss # ============================================================================================
            with tf.variable_scope('Loss'):
                net_beta = tf.placeholder(tf.float32, shape=[1], name='Beta')
                net_loss_kld = 1 + net_enc_log_var - tf.square(net_enc_mu) - tf.exp(net_enc_log_var)
                net_loss_kld = - net_beta * 0.5 * tf.reduce_sum(net_loss_kld, reduction_indices=1)
                net_loss_lkh = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=net_dec_h4, labels=self._net_input), reduction_indices=1)
                self._net_train_step = tf.constant([1.])
                self._net_loss = tf.reduce_mean(net_loss_kld + net_loss_lkh)

            # Define Accuracy # ========================================================================================
            with tf.variable_scope('Performance'):
                self._net_performance = self._net_loss

            # Define Optimizer # =======================================================================================
            with tf.variable_scope('Optimizer'):
                net_optimizer = tf.train.AdamOptimizer(lr)
                self._net_gradients = net_optimizer.compute_gradients(self._net_loss)
                self._net_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self._net_gradients]
                self._net_optimize = net_optimizer.apply_gradients(self._net_gradients)

            # Define Train Dict # ======================================================================================
            self._net_train_dict = {net_beta:[beta], net_is_training:True}
            self._net_test_dict = {net_beta:[beta], net_is_training:False}

            # Define Summaries # =======================================================================================
            with tf.variable_scope("Summaries"):
                with tf.variable_scope("Scalars"):
                    tf.summary.scalar('Loss', self._net_loss)
                    tf.summary.scalar('Kullback-Liebler', tf.reduce_mean(net_loss_kld))
                    tf.summary.scalar('LogLikelihood', tf.reduce_mean(-net_loss_lkh))
                    tf.summary.scalar('Performance', self._net_performance)


class DenseAE(BaseNetwork):
    """
    A Deep Dense AutoEncoder following specifications from Irina Higgins's paper. Trained using direct simple Gradient
    descent on full stack.
    """

    def _construct_arch(self, lr=1e-4, emb_size=10, signal_size=70**2, bn=False):

        with self._tf_graph.as_default():

            # Define Network # =========================================================================================

            with tf.variable_scope('Input'):
                self._net_input = tf.placeholder(tf.float32, shape=[None, signal_size], name='Input')
                net_is_training = tf.placeholder(tf.bool, name="IsTraining")
            with tf.variable_scope("Encoder"):
                with tf.variable_scope("DenseLayer_1"):
                    net_enc_w1 = tf.Variable(tf.truncated_normal([signal_size, 1200], stddev=0.001), name="Weights")
                    net_enc_b1 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_enc_h1 = tf.nn.relu(tf.matmul(self._net_input, net_enc_w1) + net_enc_b1)
                    if bn: net_enc_h1 = batch_norm(net_enc_h1, is_training = net_is_training)
                with tf.variable_scope("DenseLayer_2"):
                    net_enc_w2 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_enc_b2 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_enc_h2 = tf.nn.relu(tf.matmul(net_enc_h1, net_enc_w2) + net_enc_b2)
                    if bn: net_enc_h2 = batch_norm(net_enc_h2, is_training = net_is_training)
                with tf.variable_scope("DenseLayer_3"):
                    net_enc_w3 = tf.Variable(tf.truncated_normal([1200, emb_size], stddev=0.001), name="Weights")
                    net_enc_b3 = tf.Variable(tf.constant(0., shape=[emb_size]), name="Biases")
                    net_enc_z = tf.nn.sigmoid(tf.matmul(net_enc_h2, net_enc_w3) + net_enc_b3)
                    if bn: net_enc_z = batch_norm(net_enc_z, is_training = net_is_training)
                    self._net_latent = tf.stack([net_enc_z, net_enc_z], axis=1)
            with tf.variable_scope("Decoder"):
                with tf.variable_scope("DenseLayer_1"):
                    net_dec_w1 = tf.Variable(tf.truncated_normal([emb_size, 1200], stddev=0.001), name="Weights")
                    net_dec_b1 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_h1 = tf.nn.relu(tf.matmul(net_enc_z, net_dec_w1) + net_dec_b1)
                    if bn: net_dec_h1 = batch_norm(net_dec_h1, is_training = net_is_training)
                with tf.variable_scope("DenseLayer_2"):
                    net_dec_w2 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_dec_b2 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_h2 = tf.nn.relu(tf.matmul(net_dec_h1, net_dec_w2) + net_dec_b2)
                    if bn: net_dec_h2 = batch_norm(net_dec_h2, is_training = net_is_training)
                """
                with tf.variable_scope("DenseLayer_3"):
                    net_dec_w3 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_dec_b3 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_h3 = tf.nn.tanh(tf.matmul(net_dec_h2, net_dec_w3) + net_dec_b3)
                    if bn: net_dec_h3 = batch_norm(net_dec_h3, is_training=net_is_training)
                """
                with tf.variable_scope("OutputLayer_4"):
                    net_dec_w4 = tf.Variable(tf.truncated_normal([1200, signal_size], stddev=0.001), name="Weights")
                    net_dec_b4 = tf.Variable(tf.constant(0., shape=[signal_size]), name="Biases")
                    net_dec_h4 = tf.matmul(net_dec_h2, net_dec_w4) + net_dec_b4
            with tf.variable_scope('Output'):
                self._net_output = tf.nn.sigmoid(net_dec_h4, name='Output')
                self._net_label = tf.placeholder(tf.float32, shape=[None, signal_size], name='Label')

            # Define Loss # ============================================================================================
            with tf.variable_scope('Loss'):
                net_loss_lkh = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=net_dec_h4, labels=self._net_input), reduction_indices=1)
                self._net_loss = tf.reduce_mean(net_loss_lkh)
                self._net_train_step = tf.constant([1.])

            # Define Accuracy # ========================================================================================
            with tf.variable_scope('Performance'):
                self._net_performance = self._net_loss

            # Define Optimizer # =======================================================================================
            with tf.variable_scope('Optimizer'):
                net_optimizer = tf.train.AdagradOptimizer(lr)
                self._net_gradients = net_optimizer.compute_gradients(self._net_loss)
                self._net_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self._net_gradients]
                self._net_optimize = net_optimizer.apply_gradients(self._net_gradients)

            # Define Train Dict # ======================================================================================
            self._net_train_dict = {net_is_training:True}
            self._net_test_dict = {net_is_training:False}

            # Define Summaries # ======================================================================================
            with tf.variable_scope('Summaries'):
                with tf.variable_scope('Scalars'):
                    tf.summary.scalar('Loss', self._net_loss)
                    tf.summary.scalar('LogLikelihood', tf.reduce_mean(-net_loss_lkh))
                    tf.summary.scalar('Performance', self._net_performance)

class DenseSDAE(BaseNetwork):
    """
    A Simple Dense Stacked Denoising AutoEncoder trained with Greedy Layer Wise PreTraining.
    """

    def _construct_arch(self, lr=1e-4, emb_size=10, signal_size=70**2, bn=False, pre_duration=2e4, noise_lvl=0.5):

        with self._tf_graph.as_default():

            # Define Network # =========================================================================================

            with tf.variable_scope('Input'):
                self._net_input = tf.placeholder(tf.float32, shape=[None, signal_size], name='Input')
                noise = tf.multinomial(tf.log([[noise_lvl, 1 - noise_lvl]]), tf.size(self._net_input))
                noise = tf.cast(tf.reshape(noise, shape=tf.shape(self._net_input)), dtype=tf.float32)
                net_noised_input = self._net_input*noise
                self._net_train_step = tf.constant([1.])
                net_is_training = tf.placeholder(tf.bool, name="IsTraining")
            with tf.variable_scope("Encoder"):
                with tf.variable_scope("DenseLayer_1"):
                    net_enc_w1 = tf.Variable(tf.truncated_normal([signal_size, 1200], stddev=0.001), name="Weights")
                    net_enc_b1 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_enc_h1 = tf.nn.sigmoid(tf.matmul(net_noised_input, net_enc_w1) + net_enc_b1)
                    if bn: net_enc_h1 = batch_norm(net_enc_h1, is_training = net_is_training)
                with tf.variable_scope("DenseLayer_2"):
                    net_enc_w2 = tf.Variable(tf.truncated_normal([1200, 256], stddev=0.001), name="Weights")
                    net_enc_b2 = tf.Variable(tf.constant(0., shape=[256]), name="Biases")
                    net_enc_h2 = tf.nn.sigmoid(tf.matmul(net_enc_h1, net_enc_w2) + net_enc_b2)
                    if bn: net_enc_h2 = batch_norm(net_enc_h2, is_training = net_is_training)
                with tf.variable_scope("DenseLayer_3"):
                    net_enc_w3 = tf.Variable(tf.truncated_normal([256, emb_size], stddev=0.001), name="Weights")
                    net_enc_b3 = tf.Variable(tf.constant(0., shape=[emb_size]), name="Biases")
                    net_enc_z = tf.nn.sigmoid(tf.matmul(net_enc_h2, net_enc_w3) + net_enc_b3)
                    if bn: net_enc_z = batch_norm(net_enc_z, is_training = net_is_training)
                    self._net_latent = tf.stack([net_enc_z, net_enc_z], axis=1)
            with tf.variable_scope("Decoder"):
                with tf.variable_scope("DenseLayer_1"):
                    net_dec_w1 = tf.Variable(tf.truncated_normal([emb_size, 256], stddev=0.001), name="Weights")
                    net_dec_b1 = tf.Variable(tf.constant(0., shape=[256]), name="Biases")
                    net_dec_h1 = tf.nn.tanh(tf.matmul(net_enc_z, net_dec_w1) + net_dec_b1)
                    if bn: net_dec_h1 = batch_norm(net_dec_h1, is_training = net_is_training)
                with tf.variable_scope("DenseLayer_2"):
                    net_dec_w2 = tf.Variable(tf.truncated_normal([256, 1200], stddev=0.001), name="Weights")
                    net_dec_b2 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_bp2 = tf.cond((self._net_train_step[0]/(1.*pre_duration))<1., lambda: tf.constant(1.), lambda: tf.constant(0.))
                    net_dec_h2 = (1.-net_dec_bp2)*net_dec_h1 + net_dec_bp2*net_enc_h2
                    net_dec_h2 = tf.nn.tanh(tf.matmul(net_dec_h2, net_dec_w2) + net_dec_b2)
                    if bn: net_dec_h2 = batch_norm(net_dec_h2, is_training = net_is_training)
                with tf.variable_scope("DenseLayer_3"):
                    net_dec_w3 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.001), name="Weights")
                    net_dec_b3 = tf.Variable(tf.constant(0., shape=[1200]), name="Biases")
                    net_dec_bp3 = tf.cond((self._net_train_step[0]/(0.5*pre_duration))<1., lambda: tf.constant(1.), lambda: tf.constant(0.))
                    net_dec_h3 = (1.-net_dec_bp3)*net_dec_h2 + net_dec_bp3*net_enc_h1
                    net_dec_h3 = tf.nn.tanh(tf.matmul(net_dec_h3, net_dec_w3) + net_dec_b3)
                    if bn: net_dec_h3 = batch_norm(net_dec_h3, is_training=net_is_training)
                with tf.variable_scope("OutputLayer_4"):
                    net_dec_w4 = tf.Variable(tf.truncated_normal([1200, signal_size], stddev=0.001), name="Weights")
                    net_dec_b4 = tf.Variable(tf.constant(0., shape=[signal_size]), name="Biases")
                    net_dec_h4 = tf.matmul(net_dec_h3, net_dec_w4) + net_dec_b4
            with tf.variable_scope('Output'):
                self._net_output = tf.nn.sigmoid(net_dec_h4, name='Output')
                self._net_label = tf.placeholder(tf.float32, shape=[None, signal_size], name='Label')

            # Define Loss # ============================================================================================
            with tf.variable_scope('Loss'):
                net_loss_lkh = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=net_dec_h4, labels=self._net_input), reduction_indices=1)
                self._net_loss = tf.reduce_mean(net_loss_lkh)

            # Define Accuracy # ========================================================================================
            with tf.variable_scope('Performance'):
                self._net_performance = self._net_loss

            # Define Optimizer # =======================================================================================
            with tf.variable_scope('Optimizer'):
                net_optimizer = tf.train.AdamOptimizer(lr)
                self._net_gradients = net_optimizer.compute_gradients(self._net_loss)
                self._net_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self._net_gradients]
                self._net_optimize = net_optimizer.apply_gradients(self._net_gradients)

            # Define Train Dict # ======================================================================================
            self._net_train_dict = {net_is_training:True}
            self._net_test_dict = {net_is_training:False}

            # Define Summaries # ======================================================================================
            with tf.variable_scope('Summaries'):
                with tf.variable_scope('Scalars'):
                    tf.summary.scalar('Loss', self._net_loss)
                    tf.summary.scalar('LogLikelihood', tf.reduce_mean(-net_loss_lkh))
                    tf.summary.scalar('Performance', self._net_performance)
                    tf.summary.scalar('Bypass2', net_dec_bp2)
                    tf.summary.scalar('Bypass3', net_dec_bp3)
