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
optimizer=adadelta  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
encoder='lstm'  # TODO: can be removed must be lstm.
saveto='lstm_model.npz'  # The best model will be saved there
validFreq=370  # Compute the validation error after this number of update.
saveFreq=1110  # Save the parameters after every saveFreq updates
maxlen=100  # Sequence longer then this get ignored
batch_size=16  # The batch size during training.
valid_batch_size=64  # The batch size used for validation/test set.
dataset='imdb'
noise_std=0.,
use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
reload_model=None,  # Path to a saved model we want to start from.
test_size=-1,  # If >0, we keep only this number of test example.

def get_random_minibatches_index(num_training_data, _batch_size=batch_size):
    index_list=np.arange(num_training_data)
    return np.random.shuffle(index_list)[:_batch_size]


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


def _step(m_, x_, h_, c_):
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
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def create_tf_model_parameters():
    lstm_W = np.concatenate([ortho_weight(dim_proj),
                             ortho_weight(dim_proj),
                             ortho_weight(dim_proj),
                             ortho_weight(dim_proj)], axis=1)


    lstm_U = np.concatenate([ortho_weight(dim_proj),
                             ortho_weight(dim_proj),
                             ortho_weight(dim_proj),
                             ortho_weight(dim_proj)], axis=1)

    lstm_b = np.zeros((4 * dim_proj,))

    # embedding
    word_embedding = (0.01 * np.random.rand(n_words, dim_proj)).astype('float32')

    return tf.Vairable(lstm_U), tf.Variable(lstm_U), tf.Variable(lstm_b), tf.Variable(word_embedding)

class lstm_model(object):
    self._x = tf.placeholder(tf.int64, shape=(batch_size, batch_size))
    self._mask = tf.placeholder(tf.float32, shape=(batch_size, batch_size))
    self._y = tf.placeholder(tf.float32, shape=(batch_size))

    h_0 = tf.Variable(tf.zeros([n_samples, dim_proj]), tf.float32)
    c_0 = tf.Variable(tf.zeros([n_samples, dim_proj]), tf.float32)
    h_out = h_0[:, :, None]
    self._x = tf.matmul(self._x, lstm_W) + lstm_b
    # traverse through the recurring layers
    for t in n_timesteps:
        h_ = h_0
        c_ = c_0
        (h, c) = _step(mask[t, :, :], inputs[t, :, :], h_, c_)
        h_out = np.concatenate([h_out, h[:, :, None]])
        h_ = h
        c_ = c
    # accumulate along each sentence
    h_out = (h_out * mask[:, :, None]).sum(axis=0)  # resulting dim is (n_samples x n_timesteps)
    # mean pooling
    h_out = h_out / mask.sum(axis=0)[:, None]  # h_out is (n_samples x n_timesteps)

    softmax_probabilities = tf.nn.softmax(tf.matmul(h_out, lstm_U) + lstm_b)
    predictions = tf.argmax(softmax_probabilities,dimension=1)

    offset = 1e-8
    # we add an offset to the probabilities so that the log wouldn't return something to negative ?
    # what's the difference between using mean() instead of using sum()
    cost = -tf.log(softmax_probabilities[np.arange(n_samples),y]+offset).mean()

    optimizer=tf.train.AdadeltaOptimizer(learning_rate=lrate)
    tvars=tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    train_op = optimizer.apply_gradients(zip(grads,tvars))


int main():

    train, valid, test = load_data(n_words=n_words, validation_portion=0.05, maxlen=maxlen)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = np.arange(len(test[0]))
        np.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    total_num_training_data = len(train[0])
    with tf.Graph().as_default(), tf.Session() as session:
        lstm_U, lstm_U, lstm_b, word_embedding = create_tf_model_parameters()


    




        training_index = get_random_minibatches_index(total_num_training_data, batch_size)
        x = train[0][training_index]
        y = train[1][training_index]
        x, mask, y = prepare_data(x, y)

        n_timesteps = x.shape[0]
        n_samples   = x.shape[1]

        # inputs is a 3-tensor
        inputs = word_embedding[x.flatten()].reshape([n_timesteps, n_samples, dim_proj]])





















if __name__=='__main__':
    main()









