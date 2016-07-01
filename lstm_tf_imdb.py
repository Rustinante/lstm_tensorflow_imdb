# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from imdb import *

GPU_ID=1
dim_proj=128
vocabulary_size = 10000
BATCH_SIZE=16

class Options(object):
    m_proj = 128
    patience = 10
    max_epoch = 5000
    display_frequency = 10
    decay_c = 0.  # Weight decay for the classifier applied to the U weights.
    lrate = 0.0001  # Learning rate for sgd (not used for adadelta and rmsprop)
    vocabulary_size = 10000  # Vocabulary size
    # optimizer=adadelta  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder = 'lstm'  # TODO: can be removed must be lstm.
    saveto = 'lstm_model.npz'  # The best model will be saved there
    validFreq = 370  # Compute the validation error after this number of update.
    saveFreq = 1110  # Save the parameters after every saveFreq updates
    maxlen = 100  # Sequence longer then this get ignored
    valid_batch_size = 64  # The batch size used for validation/test set.
    dataset = 'imdb'
    noise_std = 0.,
    use_dropout = True,  # if False slightly faster, but worst test error
    # This frequently need a bigger model.
    reload_model = None,  # Path to a saved model we want to start from.
    test_size = -1,  # If >0, we keep only this number of test example.

    init_scale = 0.05
    learning_rate = 0.0001
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 128
    max_max_epoch = 1
    keep_prob = 1
    lr_decay = 1
    batch_size = BATCH_SIZE

config = Options()

class LSTM_Model(object):
    def __init__(self):
        #number of LSTM units, in this case it is dim_proj=128
        self.size = config.hidden_size
        #one LSTM cell with 128 units
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.size, forget_bias=0.0)
        # learning rate as a tf variable. Its value is therefore session dependent
        self._lr = tf.Variable(config.lrate, trainable=False)
        '''
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        '''

        with tf.variable_scope("RNN"):
            # initialize a word_embedding scheme out of random
            self.word_embedding = tf.get_variable('word_embedding',shape=[vocabulary_size, dim_proj],
                                                  initializer=tf.random_uniform_initializer(minval=0,maxval=0.01))
            # softmax weights and bias
            self.softmax_w = tf.get_variable("softmax_w", [dim_proj, 2], dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(0, 0.1))
            self.softmax_b = tf.get_variable("softmax_b", [2], dtype=tf.float32,
                                             initializer=tf.constant_initializer(0, tf.float32))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self._lr, lr_value))

    def create_variables(self, embedded_inputs):
        print("creating variables")
        self._initial_state = self.cell.zero_state(config.batch_size, tf.float32)
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self._targets = tf.placeholder(tf.int64, [batch_size],name='targets')
        self._mask = tf.placeholder(tf.float32, [num_steps, batch_size],name='mask')
        self.inputs = embedded_inputs

        # if is_training and config.keep_prob < 1:
        # inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)

        self.outputs = []
        state = self._initial_state
        print("in create_variables")
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = self.cell(self.inputs[time_step, :, :], state)
            self.outputs.append(cell_output)

        self.outputs = tf.reshape(tf.concat(1, self.outputs), [-1, dim_proj])
        #self.outputs now has dim (num_steps * batch_size x dim_proj)
        #each small block of the matrix is a sentence's transformed output

        # mean pooling
        # accumulate along each sentence
        print("mean pooling starts")
        segment_IDs = np.arange(batch_size).repeat(num_steps)
        pool_sum = tf.segment_sum(self.outputs, segment_ids=segment_IDs)  # pool_sum has shape (batch_size x dim_proj)

        num_words_in_each_sentence = tf.reduce_sum(self._mask, reduction_indices=0)
        tiled_num_words_in_each_sentence = tf.tile(tf.reshape(num_words_in_each_sentence, [-1, 1]), [1, dim_proj])
        pool_mean = tf.mul(pool_sum, tiled_num_words_in_each_sentence) # shape (batch_size x dim_proj)
        print("mean pooling finished")
        offset = 1e-8
        self.softmax_probabilities = tf.nn.softmax(tf.matmul(pool_mean, softmax_w) + softmax_b)

        on_value=float(1)
        off_value=float(0)
        one_hot_targets = tf.one_hot(self._targets, 2, on_value=on_value, off_value=off_value, axis=1)
        print("computing the cost")
        self._cost = -tf.reduce_mean(tf.reduce_sum(tf.log(self.softmax_probabilities + offset) * one_hot_targets,
                            reduction_indices=1))
        self.predictions = tf.argmax(self.softmax_probabilities, dimension=1)
        self.correct_predictions = tf.equal(self.predictions, self._targets)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

        print("finished computing the cost")
        self._final_state = state
        self._train_op = tf.train.AdadeltaOptimizer(learning_rate=self.lr).minimize(self._cost)

        print("finished creating variables")

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
      return self._lr

    @property
    def train_op(self):
      return self._train_op



def run_epoch(session, m, data, is_training, verbose=False, validation_data=None):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0

    #training_index = get_random_minibatches_index(len(data[0]), BATCH_SIZE)
    total_num_batches = len(data[0]) // BATCH_SIZE
    print("total number of batches is: %d" %total_num_batches)

    x      = [data[0][BATCH_SIZE * i : BATCH_SIZE * (i+1)] for i in range(total_num_batches)]
    labels = [data[1][BATCH_SIZE * i : BATCH_SIZE * (i+1)] for i in range(total_num_batches)]

    counter=0
    for mini_batch_number, (_x, _y) in enumerate(zip(x,labels)):
        counter+=1

        x_mini, mask, labels_mini, maxlen = prepare_data(_x, _y)
        embedded_inputs = words_to_embedding(m.word_embedding, x_mini)
        config.num_steps = maxlen
        print("Creating variables %d th time " %mini_batch_number)
        m.create_variables(embedded_inputs)

        print("Created variables %d th time!!! " % mini_batch_number)
        print("Initializing all variables %d th time " % mini_batch_number)
        tf.initialize_all_variables().run()
        print("Initialized all variables %d th time!!! " % mini_batch_number)
        if is_training is True:
            cost, state, _, accuracy = session.run([m.cost, m.final_state, m.train_op,m.accuracy],
                                     {m.targets: labels_mini,
                                      m.initial_state: state,
                                      m._mask: mask})
            costs += cost
            iters += m.num_steps
            print("training accuracy is: %f" %accuracy)
            '''
            if verbose and mini_batch_number % 10 == 0 and counter is not 1:
                print("VALIDATING ACCURACY\n")
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                    (mini_batch_number * 1.0 / total_num_batches, np.exp(costs / iters),
                    iters * m.batch_size / (time.time() - start_time)))

                valid_perplexity = run_epoch(session, m, validation_data, is_training=False)
                print("Epoch: %d Valid Perplexity: %.3f" % (1, valid_perplexity))
                print("finished VALIDATING ACCURACY\n")
                '''
        else:
            cost, state, accuracy = session.run([m.cost, m.final_state, m.accuracy],
                                         {m.targets: labels_mini,
                                          m.initial_state: state,
                                          m._mask: mask})
            costs += cost
            iters += m.num_steps
            print("validation/test accuracy is: %f" %accuracy)

    return np.exp(costs / iters)

def words_to_embedding(word_embedding, word_matrix):
    dim0 = word_matrix.shape[0]
    dim1 = word_matrix.shape[1]
    unrolled_matrix = tf.reshape(word_matrix,[-1])

    on_value = float(1)
    off_value = float(0)
    one_hot = tf.one_hot(indices=unrolled_matrix, depth=config.vocabulary_size, on_value=on_value, off_value=off_value, axis=1)
    embedded_words = tf.matmul(one_hot, word_embedding)
    embedded_words = tf.reshape(embedded_words, [dim0, dim1, dim_proj])
    return embedded_words

def get_random_minibatches_index(num_training_data, _batch_size=BATCH_SIZE):
    index_list=np.arange(num_training_data)
    np.random.shuffle(index_list)
    return index_list[:_batch_size]

def main():
    train_data, valid_data, test_data = load_data(n_words=vocabulary_size, validation_portion=0.05)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = LSTM_Model()

        for i in range(config.max_max_epoch):
            #lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            #m.assign_lr(session, config.learning_rate * lr_decay)
            m.assign_lr(session, config.learning_rate)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_data, is_training=True, verbose=True,validation_data=valid_data)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            #valid_perplexity = run_epoch(session, mvalid, valid_data)
            #print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        print("FINISHED TRAINING\n")
        test_perplexity = run_epoch(session, m, test_data, is_training=False)
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    main()
