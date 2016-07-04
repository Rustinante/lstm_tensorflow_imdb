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

"""
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

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from imdb import *

MAXLEN=100
np.random.seed(123)

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
    vocabulary_size = 10000  # Vocabulary size
    saveto = 'lstm_model.npz'  # The best model will be saved there
    validFreq = 370  # Compute the validation error after this number of update.
    saveFreq = 1110  # Save the parameters after every saveFreq updates
    maxlen = 100  # Sequence longer then this get ignored
    valid_batch_size = 64  # The batch size used for validation/test set.
    use_dropout = True,  # if False slightly faster, but worst test error
    # This frequently need a bigger model.
    reload_model = None,  # Path to a saved model we want to start from.
    test_size = -1,  # If >0, we keep only this number of test example.

    init_scale = 0.05
    learning_rate = 0.0001
    max_grad_norm = 5
    num_layers = 2

    hidden_size = 128
    max_max_epoch = 5000
    keep_prob = 1
    lr_decay = 1
    batch_size = BATCH_SIZE

config = Options()

class LSTM_Model(object):
    def __init__(self):
        #number of LSTM units, in this case it is dim_proj=128
        self.size = config.hidden_size
        # learning rate as a tf variable. Its value is therefore session dependent
        self._lr = tf.Variable(config.learning_rate, trainable=False)
        with tf.device("/cpu:0"):
            self._inputs = tf.placeholder(tf.float32,[MAXLEN,BATCH_SIZE],name='embedded_inputs')
            self._targets = tf.placeholder(tf.float32, [None, 2],name='targets')
            self._mask = tf.placeholder(tf.float32, [None, None],name='mask')

        def ortho_weight(ndim):
            np.random.seed(123)
            W = np.random.randn(ndim, ndim)
            u, s, v = np.linalg.svd(W)
            return u.astype(np.float32)

        with tf.variable_scope("RNN") as self.RNN_name_scope:
            # initialize a word_embedding scheme out of random
            np.random.seed(123)
            random_embedding = 0.01 * np.random.rand(10000, dim_proj)
            with tf.device("/cpu:0"):
                word_embedding = tf.get_variable('word_embedding', shape=[vocabulary_size, dim_proj],
                                                  initializer=tf.constant_initializer(random_embedding),dtype=tf.float32)

                unrolled_inputs=tf.reshape(self._inputs,[1,-1])
                embedded_inputs = tf.nn.embedding_lookup(word_embedding, unrolled_inputs)
                embedded_inputs = tf.reshape(embedded_inputs, [MAXLEN, BATCH_SIZE, dim_proj])

            # softmax weights and bias
            np.random.seed(123)
            softmax_w = 0.01 * np.random.randn(dim_proj, 2).astype(np.float32)
            softmax_w = tf.get_variable("softmax_w", [dim_proj, 2], dtype=tf.float32,
                                             initializer=tf.constant_initializer(softmax_w))
            softmax_b = tf.get_variable("softmax_b", [2], dtype=tf.float32,
                                             initializer=tf.constant_initializer(0, tf.float32))
            # cell weights and bias
            lstm_W = np.concatenate([ortho_weight(dim_proj),
                                     ortho_weight(dim_proj),
                                     ortho_weight(dim_proj),
                                     ortho_weight(dim_proj)], axis=1)

            lstm_U = np.concatenate([ortho_weight(dim_proj),
                                     ortho_weight(dim_proj),
                                     ortho_weight(dim_proj),
                                     ortho_weight(dim_proj)], axis=1)
            lstm_b = np.zeros((4 * 128,))

            lstm_W = tf.get_variable("lstm_W", shape=[dim_proj, dim_proj * 4],dtype=tf.float32,
                                          initializer=tf.constant_initializer(lstm_W))
            lstm_U = tf.get_variable("lstm_U", shape=[dim_proj, dim_proj * 4],dtype=tf.float32,
                                          initializer=tf.constant_initializer(lstm_U))
            lstm_b = tf.get_variable("lstm_b", shape=[dim_proj * 4], dtype=tf.float32, initializer=tf.constant_initializer(lstm_b))

        n_samples = BATCH_SIZE
        self.h = np.zeros([n_samples, dim_proj],dtype=np.float32)
        self.c = np.zeros([n_samples, dim_proj],dtype=np.float32)
        self.h_outputs = []

        for t in range(MAXLEN):
            mask_slice = tf.slice(self._mask, [t, 0], [1, -1])
            inputs_slice = tf.squeeze(tf.slice(embedded_inputs,[t,0,0],[1,-1,-1]))
            self.h, self.c = self.step(mask_slice,
                                       tf.matmul(inputs_slice, lstm_W) + lstm_b,
                                       self.h,
                                       self.c)
            self.h_outputs.append(tf.expand_dims(self.h, -1))

        self.h_outputs = tf.reduce_sum(tf.concat(2, self.h_outputs), 2)  # (n_samples x dim_proj)

        num_words_in_each_sentence = tf.reduce_sum(self._mask, reduction_indices=0)
        tiled_num_words_in_each_sentence = tf.tile(tf.reshape(num_words_in_each_sentence, [-1, 1]), [1, dim_proj])

        pool_mean = tf.div(self.h_outputs, tiled_num_words_in_each_sentence)
        # self.h_outputs now has dim (num_steps * batch_size x dim_proj)

        offset = 1e-8
        softmax_probabilities = tf.nn.softmax(tf.matmul(pool_mean, softmax_w) + softmax_b)
        print(tf.trainable_variables())
        print("computing the cost")
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self._targets * tf.log(softmax_probabilities), reduction_indices=1))

        self.predictions = tf.argmax(softmax_probabilities, dimension=1)
        correct_predictions = tf.equal(self.predictions, tf.argmax(self._targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        """
        grads_and_vars=opt.compute_gradients(self.cross_entropy,[self.lstm_b,
                                                                 self.lstm_W,
                                                                 self.lstm_U,
                                                                 self.word_embedding,
                                                                 softmax_w,
                                                                 softmax_b])
        self._train_op = opt.apply_gradients(grads_and_vars=grads_and_vars)
        """
        opt = tf.train.AdadeltaOptimizer(self._lr)
        self._train_op = opt.minimize(self.cross_entropy)
        print("finished computing the cost")

        '''
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        '''


    def _slice(self, x, n, dim):
        '''
        if x.ndim == 3:
            return x[:, :, n * dim : (n + 1) * dim]
            '''
        return x[:, n * dim: (n + 1) * dim]

    def step(self, mask, input, h_previous, cell_previous):
        with tf.variable_scope(self.RNN_name_scope, reuse=True):
            lstm_U = tf.get_variable("lstm_U")
        preactivation = tf.matmul(h_previous, lstm_U)
        preactivation = preactivation + input

        input_valve = tf.sigmoid(self._slice(preactivation, 0, dim_proj))
        forget_valve = tf.sigmoid(self._slice(preactivation, 1, dim_proj))
        output_valve = tf.sigmoid(self._slice(preactivation, 2, dim_proj))
        input_pressure = tf.tanh(self._slice(preactivation, 3, dim_proj))

        cell_state = forget_valve * cell_previous + input_valve * input_pressure
        cell_state = tf.tile(tf.reshape(mask, [-1, 1]), [1, dim_proj]) * cell_state + tf.tile(
            tf.reshape((1. - mask), [-1, 1]), [1, dim_proj]) * cell_previous

        h = output_valve * tf.tanh(cell_state)
        h = tf.tile(tf.reshape(mask, [-1, 1]), [1, dim_proj]) * h + tf.tile(tf.reshape((1. - mask), [-1, 1]),
                                                                            [1, dim_proj]) * h_previous
        return h, cell_state

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self._lr, lr_value))
    @property
    def cost(self):
        return self.cross_entropy
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

    print("length of the data is: %d" %len(data[0]))
    print("total number of batches is: %d" % total_num_batches)

    x      = [data[0][BATCH_SIZE * i : BATCH_SIZE * (i+1)] for i in range(total_num_batches)]
    labels = [data[1][BATCH_SIZE * i : BATCH_SIZE * (i+1)] for i in range(total_num_batches)]

    counter=0
    for mini_batch_number, (_x, _y) in enumerate(zip(x,labels)):
        counter+=1
        x_mini, mask, labels_mini, maxlen = prepare_data(_x, _y, MAXLEN_to_pad_to=MAXLEN)
        # x_mini and mask both have the shape of ( MAXLEN x BATCH_SIZE )
        #embedded_inputs = words_to_embedding(m.word_embedding, x_mini)
        print (type(x_mini[0][0]))
        if is_training is True:
            cost, accuracy, _ = session.run([m.cost ,m.accuracy,m.train_op],
                                            {m._inputs: x_mini,
                                             m._targets: labels_mini,
                                             m._mask: mask})
            costs += cost
            iters += maxlen
            print("training accuracy is: ", accuracy)
            """
            counter+=1
            if counter%50==0:
                move_on = int(raw_input("moving on? 1/0"))
                if move_on == 0:
                    break
            """

            '''
            if verbose and mini_batch_number % 10 == 0 and counter  is not 1:
                print("VALIDATING ACCURACY\n")
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                    (mini_batch_number * 1.0 / total_num_batches, np.exp(costs / iters),
                    iters * m.batch_size / (time.time() - start_time)))

                valid_perplexity = run_epoch(session, m, validation_data, is_training=False)
                print("Epoch: %d Valid Perplexity: %.3f" % (1, valid_perplexity))
                print("finished VALIDATING ACCURACY")
                '''
        '''
        else:
            cost, accuracy = session.run([m.cost, m.accuracy],
                                         {m.targets: labels_mini,
                                          m._mask: mask})
            costs += cost
            iters += maxlen

            print("validation/test accuracy is: %f" %accuracy)
        '''


    return np.exp(costs / iters)

def words_to_embedding(word_embedding, word_matrix):
    maxlen = word_matrix.shape[0]
    n_samples = word_matrix.shape[1]
    print("in words_to_embedding, maxlen= %d , n_samples= %d" %(maxlen ,n_samples))

    unrolled_matrix = np.reshape(word_matrix,[-1])
    dim0 = maxlen * n_samples
    one_hot=np.zeros((dim0, vocabulary_size),dtype=np.float32)
    for i in range(dim0):
        one_hot[i, int(unrolled_matrix[i])] = 1
    '''
    on_value = float(1)
    off_value = float(0)
    one_hot = tf.one_hot(indices=unrolled_matrix, depth=config.vocabulary_size, on_value=on_value, off_value=off_value, axis=1)
    '''
    embedded_words = tf.matmul(one_hot, word_embedding)
    embedded_words = tf.reshape(embedded_words, [maxlen, n_samples, dim_proj])
    print("embedded_words has dimension = (%d x %d x %d) "%(maxlen, n_samples, dim_proj))
    return embedded_words

def get_random_minibatches_index(num_training_data, _batch_size=BATCH_SIZE):
    index_list=np.arange(num_training_data)
    np.random.shuffle(index_list)
    return index_list[:_batch_size]

def main():
    train_data, valid_data, test_data = load_data(n_words=vocabulary_size, validation_portion=0.05,maxlen=MAXLEN)
    #with tf.Graph().as_default(), tf.Session() as session:

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with session.as_default():
        m = LSTM_Model()
        for i in range(config.max_max_epoch):
            session.run(tf.initialize_all_variables())
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
