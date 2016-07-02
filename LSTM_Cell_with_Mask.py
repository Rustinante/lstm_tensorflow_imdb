import tensorflow as tf
import numpy as np

def slice(x, n, dim):
    if x.ndim == 3:
        return x[:, :, n * dim : (n + 1) * dim]
    return x[:, n * dim : (n + 1) * dim]

def step(mask, input , h_previous, cell_previous):
    with tf.variable_scope("RNN",reuse=True)
        lstm_U = tf.get_variable("lstm_U")
    preactivation = tf.matmul(h_previous, lstm_U)
    preactivation = preactivation + input

    input_valve = tf.sigmoid(slice(preactivation, 0, dim_proj))
    forget_valve = tf.sigmoid(slice(preactivation, 1, dim_proj))
    output_valve = tf.sigmoid(slice(preactivation, 2, dim_proj))
    input_pressure = tf.tanh(slice(preactivation, 3, dim_proj))

    cell_state = forget_valve * cell_previous + input_valve * input_pressure
    cell_state = mask[:, None] * cell_state + (1. - mask)[:, None] * cell_previous

    h = output_valve * tf.tanh(cell_state)
    h = mask[:, None] * h + (1. - mask)[:, None] * h_previous

    return h, cell_state












