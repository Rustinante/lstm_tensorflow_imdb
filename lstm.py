import tensorflow as tf
import numpy as np

from imdb import *


dim_proj=128
patience=10
max_epoch=5000
display_frequency=10
decay_c=0. # Weight decay for the classifier applied to the U weights.
lrate=0.0001  # Learning rate for sgd (not used for adadelta and rmsprop)
n_words=10000  # Vocabulary size
#optimizer=adadelta  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
encoder='lstm'  # TODO: can be removed must be lstm.
saveto='lstm_model.npz'  # The best model will be saved there
validFreq=370  # Compute the validation error after this number of update.
saveFreq=1110  # Save the parameters after every saveFreq updates
maxlen=100  # Sequence longer then this get ignored
BATCH_SIZE =16  # The batch size during training.
valid_batch_size=64  # The batch size used for validation/test set.
dataset='imdb'
noise_std=0.,
use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
reload_model=None,  # Path to a saved model we want to start from.
test_size=-1,  # If >0, we keep only this number of test example.

def get_random_minibatches_index(num_training_data, _batch_size=BATCH_SIZE):
    index_list=np.arange(num_training_data)
    np.random.shuffle(index_list)
    return index_list[:_batch_size]


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


def step(m_, x_, h_, c_):
    preact = tf.matmul(h_, lstm_U)
    preact = preact + x_

    i = tf.sigmoid(_slice(preact, 0, dim_proj))
    f = tf.sigmoid(_slice(preact, 1, dim_proj))
    o = tf.sigmoid(_slice(preact, 2, dim_proj))
    c = tf.tanh(_slice(preact, 3, dim_proj))

    # c_ is the memory of the previous cell

    c = f * c_ + i * c
    c = m_[:, None] * c + (1. - m_)[:, None] * c_

    h = o * tf.tanh(c)
    h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h, c

def orthonormal_weights(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


def create_tf_model_parameters():
    lstm_W = np.concatenate([orthonormal_weights(dim_proj),
                             orthonormal_weights(dim_proj),
                             orthonormal_weights(dim_proj),
                             orthonormal_weights(dim_proj)], axis=1)
    lstm_U = np.concatenate([orthonormal_weights(dim_proj),
                             orthonormal_weights(dim_proj),
                             orthonormal_weights(dim_proj),
                             orthonormal_weights(dim_proj)], axis=1)
    lstm_b = np.zeros(dim_proj)
    # embedding
    word_embedding = (0.01 * np.random.rand(n_words, dim_proj)).astype('float32')
    return (tf.Variable(lstm_W,name='lstm_W'), tf.Variable(lstm_U,name='lstm_U'),
           tf.Variable(lstm_b,name='lstm_b'), tf.Variable(word_embedding, name='word_embedding'))

class lstm_model(object):
    def __init__(self, maxlen , batch_size, lstm_W, lstm_U, lstm_b):
        # feed into these three
        self._embedded_word_inputs= tf.placeholder(tf.float32, shape=(maxlen, batch_size, dim_proj), name="embbed_word_inputs")
        self._mask = tf.placeholder(tf.float32, shape=(batch_size, dim_proj), name="mask")
        self._labels = tf.placeholder(tf.float32, shape=(batch_size), name="labels")

        self.h_0 = tf.zeros([batch_size, dim_proj], dtype=tf.float32)
        self.c_0 = tf.zeros([batch_size, dim_proj], dtype=tf.float32)

        self.h =self.h_0
        self.c =self.c_0

        self._embedded_word_inputs = tf.reshape(self._embedded_word_inputs, [-1, dim_proj])
        # shape is ( batch_size * maxlen x dim_proj)
        self._transformed_embedded_word_inputs=tf.matmul(self._embedded_word_inputs,lstm_W)+lstm_b
        # traverse through the recurring layers
        h_out = []

        for t in range(maxlen):
            self.h, self.c = step(tf.slice(self._mask,[t,0],[1,-1]),
                                  tf.slice(self._transformed_embedded_word_inputs,[t,0,0],[1,-1,-1]),
                                  self.h, self.c)
            h_out.append(self.h)

        h_out=tf.concat(1,h_out)
        tiled_mask=tf.tile(self._mask.reshape(-1),[1,dim_proj])
        h_out=tf.mul(h_out,tiled_mask)
        # accumulate along each sentence
        segment_IDs= np.arange(batch_size).repeat(maxlen)
        pool_sum = tf.segment_sum(h_out,segment_ids=segment_IDs) # h_out now has shape (batch_size x dim_proj)
        # mean pooling
        num_words_in_each_sentence = tf.reduce_sum(self._mask, reduction_indices=0)
        tiled_num_words_in_each_sentence = tf.tile(tf.reshape(num_words_in_each_sentence,[-1,1]),[1,dim_proj])
        pool_mean = tf.mul(pool_sum,tiled_num_words_in_each_sentence)

        self.softmax_probabilities = tf.nn.softmax(tf.matmul(pool_mean, lstm_U) + lstm_b)
        self.predictions = tf.argmax(self.softmax_probabilities,dimension=1)

        offset = 1e-8
        # we add an offset to the probabilities so that the log wouldn't return something to negative ?
        # what's the difference between using mean() instead of using sum()
        self.cost = -tf.log(self.softmax_probabilities[np.arange(batch_size),y]+offset).mean()

        self.optimizer=tf.train.AdadeltaOptimizer(learning_rate=lrate)
        self.tvars=tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.tvars),
                                          config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(self.grads,self.tvars))


    def train(self, session, embedded_word_inputs, mask, labels):
        # (maxlen x batch_size x dim_proj) (batch_size x dim_proj) (batch_size)
        cost, _ = session.run([self.cost, self.train_op],
                    feed_dict={
                    self._embedded_word_inputs: embedded_word_inputs,
                    self._mask : mask,
                    self._labels : labels})
        return cost


def main():
    train, valid, test = load_data(n_words=n_words, validation_portion=0.05)
    '''
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = np.arange(len(test[0]))
        np.random.shuffle(idx)
        idx = idx[0:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
    '''
    total_num_training_data = len(train[0])

    with tf.Graph().as_default(), tf.Session() as session:
        lstm_W, lstm_U, lstm_b, word_embedding = create_tf_model_parameters()
        training_index = get_random_minibatches_index(total_num_training_data, BATCH_SIZE)
        x = [train[0][i] for i in training_index]
        labels = [train[1][i] for i in training_index]
        x, mask, labels, maxlen = prepare_data(x, labels)

        maxlen = x.shape[0]
        n_samples   = x.shape[1]
        # inputs is a 3-tensor

        tf.slice(word_embedding,[x.flatten()],[])

        inputs = word_embedding[x.flatten()].reshape([maxlen, n_samples, dim_proj])
        model = lstm_model(maxlen , BATCH_SIZE, lstm_W, lstm_U, lstm_b)
        cost = model.train(session, embedded_word_inputs, mask, labels)
        print("the cost is: %f " %cost)





if __name__=='__main__':
    main()









