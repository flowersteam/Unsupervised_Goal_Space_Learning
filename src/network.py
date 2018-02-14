#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains base implementation of a NN classifier trained using supervised learning.

Author: Alexandre Péré

"""

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from tensorflow.contrib.tensorboard.plugins import projector
import numpy
import os
import pickle
import sys
if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class BaseNetwork(object):
    """
    This defines the basic network structure we use.
    """

    def __init__(self, logs_name=None, path_to_logs=os.getcwd(), **kwargs):
        """
        The initializer of the BasicNetwork object.
        """

        # Initialize super
        object.__init__(self)

        # We initialize the variables of the object
        self._tf_graph = tf.Graph()
        self._tf_session =None
        self._tf_fw = None
        self._net_loss = None
        self._net_train_step = None
        self._net_optimize = None
        self._net_input = None
        self._net_label = None
        self._net_output = None
        self._net_latent = None
        self._net_gradients = None
        self._net_weights = None
        self._net_performance = None
        self._net_train_dict = dict()
        self._net_test_dict = dict()
        self._net_summaries = None
        self._net_summaries_history = list()
        self._net_summary_parser = summary_pb2.Summary()
        self._logs_path = path_to_logs
        self._logs_name = logs_name

        # We clear logs
        os.system("rm -rf %s/*"%os.path.join(self._logs_path, self._logs_name))

        # We construct and initialize everything
        self._construct_arch(**kwargs)
        self._initialize_session()
        self._initialize_summaries()
        self._initialize_weights()

    def train(self, X_train, y_train,
              iterations=0,
              batch_size=100,
              callback=None,
              disable_progress=False):
        """
        The public training method. A network can be trained for a specified number of iterations using the _iterations_
        parameter.

        Parameters:
            + X_train: a numpy array containing training input data
            + y_train: a numpy array containing training output classes
            + iterations: number of iterations to perform
            + batch_size: the batch size for training data
            + callback: a method to be called before each printing iteration
        """

        # We check that the number of iterations set is greater than 100 if iterations is used
        if iterations<100:
            raise Warning("Number of iterations must be superior to 100")

        # We initialize filewriter
        self._initialize_fw()

        # We loop
        for iter in tqdm(range(iterations),desc="Training Model", disable=disable_progress):
            # We get the random indexes to use in the batch
            train_idx = numpy.random.permutation(X_train.shape[0])
            train_idx = train_idx[0:batch_size]
            # We execute the gradient descent step
            input_dict = {self._net_input: X_train[train_idx],
                          self._net_label: y_train[train_idx],
                          self._net_train_step: [iter]}
            input_dict.update(self._net_train_dict)
            self._net_optimize.run(feed_dict=input_dict, session=self._tf_session)
            # If the iteration is a multiple of 100, we do things
            if (iter % 100 == 0) and (iter > 0):
                # We update tensorboard summaries
                summary = self._net_summaries.eval(feed_dict=input_dict,session=self._tf_session)
                self._net_summary_parser.ParseFromString(summary)
                self._net_summaries_history.append({str(val.tag):val.simple_value for val in self._net_summary_parser.value})
                self._tf_fw.add_summary(summary,iter)
                self._tf_fw.flush()
                # We execute the callback if it exists
                if callback is not None: callback(self)

    def train_with_generator(self, gen_func, iterations=0, batch_size=100, callback=None, disable_progress=False):
        """
        The public training method using a data generation method. A network can be trained for a specified number of iterations using the _iterations_
        parameter.

        Parameters:
            + gen_func: the data generation function, taking no arguments, and generate a single data sample. It returns a tuple of two numpy arrays, of according size.
            + iterations: number of iterations to perform
            + batch_size: the batch size for training data
            + callback: a method to be called before each printing iteration
        """

        # We check that the number of iterations set is greater than 100 if iterations is used
        if iterations<100:
            raise Warning("Number of iterations must be superior to 100")

        # We initialize filewriter
        self._initialize_fw()

        # We gather dimensions of data
        x, y = gen_func()

        # We loop
        for iter in tqdm(range(iterations),desc="Training Model", disable=disable_progress):
            # We generate the batch
            X = numpy.repeat(numpy.expand_dims(x, 0), repeats=batch_size, axis=0)
            Y = numpy.repeat(numpy.expand_dims(y, 0), repeats=batch_size, axis=0)
            for i in range(batch_size):
                X[i], Y[i] = gen_func()
            # We execute the gradient descent step
            input_dict = {self._net_input: X,
                          self._net_label: Y,
                          self._net_train_step: [iter]}
            input_dict.update(self._net_train_dict)
            self._net_optimize.run(feed_dict=input_dict, session=self._tf_session)
            # If the iteration is a multiple of 100, we do things
            if (iter % 100 == 0) and (iter > 0):
                # We update tensorboard summaries
                summary = self._net_summaries.eval(feed_dict=input_dict,session=self._tf_session)
                self._net_summary_parser.ParseFromString(summary)
                self._net_summaries_history.append({str(val.tag):val.simple_value for val in self._net_summary_parser.value})
                self._tf_fw.add_summary(summary,iter)
                self._tf_fw.flush()
                # We execute the callback if it exists
                if callback is not None: callback(self)

    def evaluate_output(self, X, disable_progress=False):
        """
        The public output evaluation method.

        Parameters:
            + X: a numpy array containing input data

        Returns:
            + a numpy array containing the evaluations
        """

        # We instantiate the output array
        out_arr = list()

        # We loop through the samples to evaluate the network value
        for iter in tqdm(range(0, X.shape[0]), desc="Evaluating Output", disable=disable_progress):
            input_dict = {self._net_input: X[iter:iter+1]}
            input_dict.update(self._net_test_dict)
            out_arr.append(self._net_output.eval(feed_dict=input_dict, session=self._tf_session))

        return numpy.asarray(out_arr)

    def evaluate_latent(self, X, disable_progress=False):
        """
        The public latent embedding evaluation method.

        Parameters:
            + X: a numpy array containing input data
            + disable_progress: a boolean to disable the progress bar

        Returns:
            + a numpy array containing the evaluations
        """

        # We instantiate the output array
        out_arr = list()

        # We loop through the samples to evaluate the network value
        for iter in tqdm(range(0, X.shape[0]), desc="Evaluating Output", disable=disable_progress):
            input_dict = {self._net_input: X[iter:iter + 1]}
            input_dict.update(self._net_test_dict)
            out_arr.append(self._net_latent.eval(feed_dict=input_dict, session=self._tf_session).squeeze())

        return numpy.asarray(out_arr)

    def evaluate_tensor(self, name, initial_dict='train', update_dict=None):
        """
        The public tensor evaluation method. You can eval any tensor given an input dict. The initial dict
        is basically fixed to be the train dict.

        Parameters:
            + name: the name of the tensor to evaluate
            + initial_dict: 'train' to use train_dict as initial dict, 'test' to use test dict as initial dict
            + update_dict: some input dict of your own to update the initial_dict

        Returns:
            + a numpy array containing the evaluations
        """

        # We retrieve the tensor by name
        tensor_to_eval = self.get_tensor(name)

        # We set the input dict
        if initial_dict=='train':
            input_dict = self._net_train_dict
        elif initial_dict=='test':
            input_dict = self._net_test_dict
        if update_dict is not None:
            input_dict.update(update_dict)

        # We evaluate the tensor
        out_arr = tensor_to_eval.eval(feed_dict=input_dict, session=self._tf_session)

        return out_arr

    def update_feed_dict_value(self, key, value, which):
        """
        The public feed dict update method. Used to update the learning rate during training.

        Parameters:
            + key: the dict key to update
            + value: the dict new value
            + which: if 'test' change test dict, if 'train' change train dict, if 'both' change both
        """

        if which=="test":
            self._net_test_dict[key] = value;
        elif which=="train":
            self._net_train_dict[key] = value;
        elif which=='both':
            self._net_train_dict[key] = value;
            self._net_test_dict[key] = value;

    def save(self, path):
        """
        The public saving method, which allows to save a trained network. Tensorflow do not save on a single file, hence
        path doesn't need to have an extension.

        Parameters:
            + path: Path to files like '/tmp/model'
        """

        # We save tensorflow objects
        with self._tf_graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._tf_session, os.path.abspath(path))

    def load(self, path):
        """
        The public loading method, which allows to restore a trained network. Tensorflow do not save on a single file,
        hence path doesn't need to have an extension.

        Parameters:
             + path: Path to files like '/tmp/model'
        """

        # We load the tensorflow objects
        with self._tf_graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self._tf_session, os.path.abspath(path))
            self._tf_fw.add_graph(self._tf_graph)

    def load_weights(self, path):
        """
        The public weights loading method, which allows to load pre-trained weights in the network.

        Parameters:
            + path: Path to file or folder containing data.
        """

        # We call the specific logic contained in the architecture:

        self._load_weights(path)


    def get_summaries(self, name=None):
        """
        This public method allows to retrieve the recent summaries of the network.

        Parameters:
            + name: if the name of the summary you want to retrieve, if not given, everything is returned

        Returns:
            + a list containing merged summaries if no name is provided, and an array containing the data otherwise.
        """
        if name is None:
            return self._net_summaries_history
        else:
            length = len(self._net_summaries_history)
            array = numpy.zeros([length,1])
            for i in range(0,length):
                array[i] = self._net_summaries_history[i][name]
            return array

    def get_tensor(self, name):
        """
        This public method allows to catch a tensor by its name in the architecture.

        Parameters:
            + name: the name of the tensor ex: 'Conv1/W1:0'

        Returns:
            + The tensor
        """

        return self._tf_graph.get_tensor_by_name(name)

    def get_latent_size(self):
        """
        This public method allows to retrieve latent space size.

        Returns:
            + Latent space size
        """

        return self._net_latent.shape.as_list()[1:]

    def execute_projector(self, X, emb_idx=0):
        """
        This public method allows to generate tensorboard projection.

        Attributes:
            + X: Input Data
            + emb_idx: Index of embedding
        """

        with self._tf_graph.as_default():
            latent = self.evaluate_latent(X, disable_progress=True)
            emb = tf.Variable(latent.squeeze()[:,emb_idx,:], name="Latent")
            metadata = os.path.join(self._logs_path, 'metadata.tsv')
            with open(metadata, 'w') as metadata_file:
                for index in range(X.shape[0]):
                    metadata_file.write('%d\n' % index)
            saver = tf.train.Saver([emb])
            self._tf_session.run(emb.initializer)
            saver.save(self._tf_session, os.path.join(self._logs_path, 'latent.ckpt'))
            config = projector.ProjectorConfig()
            # One can add multiple embeddings.
            embedding = config.embeddings.add()
            embedding.tensor_name = emb.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = metadata
            # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(tf.summary.FileWriter(self._logs_path), config)

    def _initialize_weights(self):
        """
        The private weights initialization method.
        """

        with self._tf_graph.as_default():

            self._tf_session.run(tf.global_variables_initializer())

    def _initialize_summaries(self):
        """
        The summary initialization method.
        """

        with self._tf_graph.as_default():
            with tf.name_scope("Parameters"):
                with tf.name_scope("Gradients"):
                        for index, grad in enumerate(self._net_gradients):
                            tf.summary.histogram("%s"%self._net_gradients[index][1].name.split(":")[0],self._net_gradients[index])
                with tf.name_scope("Values"):
                    self._net_weights = tf.trainable_variables()
                    for index, weight in enumerate(self._net_weights):
                            tf.summary.histogram("%s"%self._net_weights[index].name.split(":")[0],self._net_weights[index])
            self._net_summaries = tf.summary.merge_all()

    def _initialize_fw(self):
        """
        The private filewriter initialization method.
        """

        if self._logs_name is None:
            dir_list = [int(elem) for elem in os.listdir(self._logs_path) if os.path.isdir(os.path.join(self._logs_path,elem))
                        and elem.isdigit()]
            if len(dir_list) == 0:
                last_run = 0
            else:
                last_run = max(dir_list)
            self._tf_fw = tf.summary.FileWriter(os.path.join(self._logs_path, str(last_run+1)), graph=self._tf_graph)
        else:
            if os.path.isdir(self._logs_name):
                os.system("rm -rf %s/*"%self._logs_name)
            self._tf_fw = tf.summary.FileWriter(os.path.join(self._logs_path, str(self._logs_name)), graph=self._tf_graph)

    def _initialize_session(self):
        """
        The private session initialization method.
        """

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._tf_session = tf.Session(graph=self._tf_graph, config=config)

    def _construct_arch(self):
        """
        The private architecture construction method. Should be reimplemented, and define the computations of the
        following attributes:
            + self._net_input: the input tf placeholder
            + self._net_output: the output layer of the network
            + self._net_label: the labels tf placeholder
            + self._net_loss: the loss used to train the network (containing weights decay)
            + self._net_optimize: the optimization method to use for training
            + self._net_accuracy: the accuracy measure used to monitor the network performance
        """

        raise NotImplementedError("Virtual Method Was called")

        # with self._tf_graph.as_default():
        #
        #     # Define Network # =======================================================================================
        #
        #     # Input # ------------------------------------------------------------------------------------------------
        #     self._net_input = tf.placeholder(tf.float32, shape=[None, "Put Here Input Dim"], name='input')
        #
        #     # Output # -----------------------------------------------------------------------------------------------
        #     self._net_output = tf.nn.softmax("Put Here output layer", name='output')
        #     self._net_label = tf.placeholder(tf.float32, shape=[None, "Put Here Output Dim"], name='label')
        #
        #     # Define Loss # ==========================================================================================
        #     self._net_loss = tf.add(cross_entropy, weights_decay)
        #
        #     # Define Optimizer # =====================================================================================
        #     self._net_optimize = tf.train.AdamOptimizer(1e-4).minimize(self._net_loss)
        #
        #     # Define Accuracy # ======================================================================================
        #     correct_prediction = tf.equal(tf.argmax(self._net_output, 1), tf.argmax(self._net_label, 1))
        #     self._net_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
