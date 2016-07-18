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



dim_proj= 128
BATCH_SIZE=16
ACCURACY_THREASHOLD= 0.95
np.random.seed(123)



class Options(object):
    NUM_UNROLLS=100
    MAXLEN = 100
    VALIDATION_PORTION = 0.05
    patience = 10
    max_epoch = 20
    decay_c = 0.  # Weight decay for the classifier applied to the U weights.
    VOCABULARY_SIZE = 10000  # Vocabulary size
    saveto = 'lstm_model.npz'  # The best model will be saved there
    saveFreq = 1110  # Save the parameters after every saveFreq updates
    valid_batch_size = 16  # The batch size used for validation/test set.
    use_dropout = True,  # if False slightly faster, but worst test error
    # This frequently need a bigger model.
    reload_model = None,  # Path to a saved model we want to start from.
    test_size = -1,  # If >0, we keep only this number of test example.

    learning_rate = 0.001
    max_grad_norm = 5
    hidden_size = 128
    keep_prob = 1
    learning_rate_decay = 1
    max_sentence_length_for_testing=100


class Flag(object):
    first_training_epoch = True
    first_validation_epoch = True
    testing_epoch = False

config = Options()
flags = Flag()

class LSTM_Model(object):
    def __init__(self, is_training=True):
        #number of LSTM units, in this case it is dim_proj=128
        self.size = config.hidden_size
        # learning rate as a tf variable. Its value is therefore session dependent
        self._lr = tf.Variable(config.learning_rate, trainable=False)
        with tf.device("/cpu:0"):
            self._inputs = tf.placeholder(tf.int32,[config.MAXLEN,BATCH_SIZE],name='embedded_inputs')
            self._targets = tf.placeholder(tf.float32, [None, 2],name='targets')
            self._mask = tf.placeholder(tf.float32, [None, None],name='mask')

        def ortho_weight(ndim):
            #np.random.seed(123)
            W = np.random.randn(ndim, ndim)
            u, s, v = np.linalg.svd(W)
            return u.astype(np.float32)

        with tf.variable_scope("RNN") as self.RNN_name_scope:
            # initialize a word_embedding scheme out of random
            #np.random.seed(123)
            with tf.device("/cpu:0"):
                random_embedding = 0.01 * np.random.rand(10000, dim_proj)
                word_embedding = tf.get_variable('word_embedding', shape=[config.VOCABULARY_SIZE, dim_proj],
                                                  initializer=tf.constant_initializer(random_embedding),dtype=tf.float32)

                unrolled_inputs=tf.reshape(self._inputs,[1,-1])
                embedded_inputs = tf.nn.embedding_lookup(word_embedding, unrolled_inputs)
                embedded_inputs = tf.reshape(embedded_inputs, [config.MAXLEN, BATCH_SIZE, dim_proj])

            # softmax weights and bias
            #np.random.seed(123)
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

        for t in range(config.MAXLEN):
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
        poo_mean = tf.nn.dropout(pool_mean, 0.5)
        offset = 1e-8
        softmax_probabilities = tf.nn.softmax(tf.matmul(pool_mean, softmax_w) + softmax_b)
        self.predictions = tf.argmax(softmax_probabilities, dimension=1)
        self.num_correct_predictions = tf.reduce_sum(tf.cast(tf.equal(self.predictions, tf.argmax(self._targets, 1)), dtype=tf.float32))
        print("Constructing graphs for cross entropy")
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self._targets * tf.log(softmax_probabilities), reduction_indices=1))
        if is_training:
            print("Trainable variables: ", tf.trainable_variables())
            self._train_op = tf.train.AdamOptimizer(0.0001).minimize(self.cross_entropy)
        print("Finished constructing the graph")


    def _slice(self, x, n, dim):
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



def run_epoch(session, m, data, is_training, verbose=True):
    if is_training not in [True,False]:
        raise ValueError("mode must be one of [True, False] but received ", is_training)

    total_cost = 0.0
    num_samples_seen= 0
    total_num_correct_predictions= 0
    list_of_training_index_list = get_random_minibatches_index(len(data[0]), BATCH_SIZE)
    total_num_batches = len(data[0]) // BATCH_SIZE
    total_num_reviews = len(data[0])
    #x      = [data[0][BATCH_SIZE * i : BATCH_SIZE * (i+1)] for i in range(total_num_batches)]
    #labels = [data[1][BATCH_SIZE * i : BATCH_SIZE * (i+1)] for i in range(total_num_batches)]
    x=[]
    labels=[]
    for l in list_of_training_index_list:
        x.append([data[0][i] for i in l])
        labels.append([data[1][i] for i in l])

    if is_training:
        if flags.first_training_epoch:
            flags.first_training_epoch= False
            print("For training, total number of reviews is: %d" % total_num_reviews)
            print("For training, total number of batches is: %d" % total_num_batches)

        for mini_batch_number, (_x, _y) in enumerate(zip(x,labels)):
            #print("mini batch: %d" %mini_batch_number)
            # x_mini and mask both have the shape of ( config.MAXLEN x BATCH_SIZE )

            x_mini, mask, labels_mini = prepare_data(_x, _y, MAXLEN_to_pad_to=config.MAXLEN)
            num_samples_seen += x_mini.shape[1]
            num_correct_predictions, _ = session.run([m.num_correct_predictions, m.train_op],
                                                     feed_dict={m._inputs: x_mini,
                                                                m._targets: labels_mini,
                                                                m._mask: mask})
            #print(m.lstm_W.eval())
            total_num_correct_predictions+= num_correct_predictions

        avg_accuracy = total_num_correct_predictions/num_samples_seen
        print("Traversed through %d samples." %num_samples_seen)
        return np.asscalar(avg_accuracy)

    else:
        if flags.first_validation_epoch or flags.testing_epoch:
            flags.first_validation_epoch= False
            flags.testing_epoch= False
            print("For validation/testing, total number of reviews is: %d" % total_num_reviews)
            print("For validation/testing, total number of batches is: %d" % total_num_batches)

        for mini_batch_number, (_x, _y) in enumerate(zip(x, labels)):
            x_mini, mask, labels_mini = prepare_data(_x, _y, MAXLEN_to_pad_to=config.MAXLEN)
            num_samples_seen += x_mini.shape[1]
            cost, num_correct_predictions = session.run([m.cost ,m.num_correct_predictions],
                                                        feed_dict={m._inputs: x_mini,
                                                                   m._targets: labels_mini,
                                                                   m._mask: mask})
            total_cost += cost
            total_num_correct_predictions += num_correct_predictions
        accuracy= total_num_correct_predictions/num_samples_seen
        print("total cost is %.4f" %total_cost)
        return np.asscalar(accuracy)


# deprecated, since this doesn't seem to use gpu?
def words_to_embedding(word_embedding, word_matrix):
    maxlen = word_matrix.shape[0]
    n_samples = word_matrix.shape[1]
    print("in words_to_embedding, maxlen= %d , n_samples= %d" %(maxlen ,n_samples))

    unrolled_matrix = np.reshape(word_matrix,[-1])
    dim0 = maxlen * n_samples
    one_hot=np.zeros((dim0, config.VOCABULARY_SIZE),dtype=np.float32)
    for i in range(dim0):
        one_hot[i, int(unrolled_matrix[i])] = 1
    '''
    on_value = float(1)
    off_value = float(0)
    one_hot = tf.one_hot(indices=unrolled_matrix, depth=config.VOCABULARY_SIZE, on_value=on_value, off_value=off_value, axis=1)
    '''
    embedded_words = tf.matmul(one_hot, word_embedding)
    embedded_words = tf.reshape(embedded_words, [maxlen, n_samples, dim_proj])
    print("embedded_words has dimension = (%d x %d x %d) "%(maxlen, n_samples, dim_proj))
    return embedded_words

def get_random_minibatches_index(num_training_data, batch_size=BATCH_SIZE, shuffle=True):
    index_list=np.arange(num_training_data,dtype=np.int32)
    if shuffle:
        np.random.shuffle(index_list)
    index_list=index_list.tolist()
    total_num_batches = num_training_data//batch_size
    result=[index_list[batch_size * i : batch_size*(i+1)] for i in range(total_num_batches)]
    return result

def main():
    train_data, validation_data, test_data = load_data(n_words=config.VOCABULARY_SIZE,
                                                       validation_portion=config.VALIDATION_PORTION,
                                                       maxlen=config.MAXLEN)
    new_test_features=[]
    new_test_labels=[]
    #right now we only consider sentences of length less than config.max_sentence_length_for_testing
    for feature, label in zip(test_data[0],test_data[1]):
        if len(feature)<config.max_sentence_length_for_testing:
            new_test_features.append(feature)
            new_test_labels.append(label)
    test_data=(new_test_features,new_test_labels)
    del new_test_features, new_test_labels

    GPU_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
    session = tf.Session(config=tf.ConfigProto(gpu_options=GPU_options))

    with session.as_default():
        with tf.variable_scope("model"):
            m = LSTM_Model()
        with tf.variable_scope("model", reuse=True):
            m_test = LSTM_Model(is_training=False)

        print("Initializing all variables")
        session.run(tf.initialize_all_variables())
        print("Initialized all variables")
        saver = tf.train.Saver()
        start_time = time.time()
        try:
            for i in range(config.max_epoch):
                epoch_number= i+1
                print("\nTraining")
                m.assign_lr(session, config.learning_rate)
                print("Epoch: %d Learning rate: %.5f" % (epoch_number, session.run(m.lr)))
                average_training_accuracy = run_epoch(session, m, train_data, is_training=True)
                print("Average training accuracy in epoch %d is: %.5f" %(epoch_number, average_training_accuracy))
                if epoch_number==20:
                    print("total time is:",time.time()-start_time)

                if epoch_number%5 == 0:
                    print("\nValidating")
                    validation_accuracy = run_epoch(session, m, validation_data, is_training=False)
                    print("Validation accuracy in epoch %d is: %.5f\n" %(epoch_number, validation_accuracy))
                    print("\nTesting")
                    flags.testing_epoch = True
                    config.MAXLEN = config.max_sentence_length_for_testing
                    testing_accuracy = run_epoch(session, m_test, test_data, is_training=False)
                    config.MAXLEN =config.NUM_UNROLLS
                    print("Testing accuracy is: %.4f" % testing_accuracy)
                    if validation_accuracy > ACCURACY_THREASHOLD:
                        print("Validation accuracy reached the threashold. Breaking")
                        break
                    if epoch_number%10 == 0:
                        path = saver.save(session,"params_at_epoch.ckpt", global_step=epoch_number )
                        print("Saved parameters to %s" %path)

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()