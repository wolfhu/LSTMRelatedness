# Peter Henderson
# ==============================================================================

"""Example TreeLSTM implementation based on Socher et al.'s
TODO: link here and citation

Much of this was taken from tensorflow

TODO: link here

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

To compile on CPU:
  bazel build -c opt tensorflow/models/rnn/ptb:ptb_word_lm
To compile on GPU:
  bazel build -c opt tensorflow --config=cuda \
    tensorflow/models/rnn/ptb:ptb_word_lm
To run:
  ./bazel-bin/.../ptb_word_lm --data_path=/tmp/simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn.ptb import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS


class RelatednessModel(object):
  """The relatedness model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._left_inputs = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._right_inputs = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=1.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    left_inputs = self._left_inputs
    right_inputs = self._right_inputs

    if is_training and config.keep_prob < 1:
      left_inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in self._left_inputs]
      right_inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in self._right_inputs]

    loutputs = []
    routputs = []
    states = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step, input_left, input_right_ in enumerate(zip(left_inputs, right_inputs)):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        # left sentence
        (lcell_output, state) = cell(input_left_, state)
        (rcell_output, state) = cell(input_right_, state)
        loutputs.append(lcell_output)
        routputs.append(rcell_output)
        states.append(state)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    # Need a simple network on top for the similarity
    logits = tf.nn.xw_plus_b(output,
                             tf.get_variable("softmax_w", [size, vocab_size]),
                             tf.get_variable("softmax_b", [vocab_size]))
    # TODO: replace this with softmax
    loss = seq2seq.sequence_loss_by_example([logits],
                                            [tf.reshape(self._targets, [-1])],
                                            [tf.ones([batch_size * num_steps])],
                                            vocab_size)
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = states[-1]

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

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
