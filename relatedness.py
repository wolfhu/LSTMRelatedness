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
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import seq2seq
from data_utils import *

class RelatednessModel(object):
  """The relatedness model."""

  def process_sentence_pair(self, lsentence_raw, rsentence_raw, session, prev_state = None):
    """ TODO: this is mad inefficient esp. without symbolic
        compiling, should really batch this
        Input sentence is just string"""
    # convert sentence into word vector array
    lsentence = convert_sentence_to_glove_vectors(lsentence_raw, self.vocabulary, self.glove_word_vectors, self.word_vec_size)
    rsentence = convert_sentence_to_glove_vectors(rsentence_raw, self.vocabulary, self.glove_word_vectors, self.word_vec_size)

    # 5 x 300
    _left_inputs = tf.placeholder(tf.float32, [len(lsentence), self.config.word_vec_size])
    _right_inputs = tf.placeholder(tf.float32, [len(rsentence), self.config.word_vec_size])

    # _targets = tf.placeholder(tf.int32)

    # Apply dropout filter
    # if self.is_training and self.config.keep_prob < 1:
    #   left_inputs = [tf.nn.dropout(input_, self.config.keep_prob) for input_ in left_inputs]
    #   right_inputs = [tf.nn.dropout(input_, self.config.keep_prob) for input_ in right_inputs]

    linputs = [ tf.reshape(i, (1, self.config.word_vec_size)) for i in tf.split(0, len(lsentence), _left_inputs)]
    rinputs = [ tf.reshape(i, (1, self.config.word_vec_size)) for i in tf.split(0, len(rsentence), _right_inputs)]

    if prev_state is None:
      prev_state = self.left_lstm_cell.zero_state(1, tf.float32)

    with tf.variable_scope("LeftLSTM"):
      loutputs, rstates = rnn.rnn(self.left_lstm_cell, linputs, initial_state=prev_state, sequence_length=len(lsentence))
    with tf.variable_scope("RightLSTM"):
      routputs, rstates = rnn.rnn(self.right_lstm_cell, rinputs, initial_state=prev_state, sequence_length=len(lsentence))

    iop = tf.initialize_all_variables()
    session.run(iop)

    # TODO: the actual loss function and relatedness softmax layer
    louts = session.run(loutputs, feed_dict = {_left_inputs : lsentence, _right_inputs : rsentence })


    # outputs at each timestep of the sentence (i.e. each word)
    print(louts)
    print(len(louts))
    # print(routs)

  def __init__(self, is_training, glove_word_vectors, vocabulary, config):
    self.size = config.hidden_size
    self.config = config
    self.is_training = is_training
    self.word_vec_size = config.word_vec_size
    vocab_size = config.vocab_size
    self.glove_word_vectors = glove_word_vectors
    self.vocabulary = vocabulary

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.

    # TODO: these might be able to be improved if used the LSTMCell which has other features
    # to improve performance, but then need the sentence_length
    with tf.variable_scope("LeftLSTM"):
        self.left_lstm_cell = rnn_cell.BasicLSTMCell(self.size, forget_bias=1.0)
    with tf.variable_scope("RightLSTM"):
        self.right_lstm_cell = rnn_cell.BasicLSTMCell(self.size, forget_bias=1.0)
    if is_training and config.keep_prob < 1:
      with tf.variable_scope("LeftLSTM"):
        self.left_lstm_cell = rnn_cell.DropoutWrapper(self.left_lstm_cell, output_keep_prob=config.keep_prob)
      with tf.variable_scope("RightLSTM"):
        self.right_lstm_cell = rnn_cell.DropoutWrapper(self.right_lstm_cell, output_keep_prob=config.keep_prob)

    with tf.variable_scope("LeftLSTM"):
      self.left_lstm_cell = rnn_cell.MultiRNNCell([self.left_lstm_cell] * config.num_layers)
    with tf.variable_scope("RightLSTM"):
      self.right_lstm_cell = rnn_cell.MultiRNNCell([self.right_lstm_cell] * config.num_layers)

    # output = tf.reshape(tf.concat(1, outputs), [-1, size])
    # # Need a simple network on top for the similarity
    # logits = tf.nn.xw_plus_b(output,
    #                          tf.get_variable("softmax_w", [size, vocab_size]),
    #                          tf.get_variable("softmax_b", [vocab_size]))
    # # TODO: replace this with softmax
    # loss = seq2seq.sequence_loss_by_example([logits],
    #                                         [tf.reshape(self._targets, [-1])],
    #                                         [tf.ones([batch_size * num_steps])],
    #                                         vocab_size)
    # self._cost = cost = tf.reduce_sum(loss) / batch_size
    # self._final_state = states[-1]
    #
    # if not is_training:
    #   return
    #
    # self._lr = tf.Variable(0.0, trainable=False)
    # tvars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
    #                                   config.max_grad_norm)
    # optimizer = tf.train.GradientDescentOptimizer(self.lr)
    # self._train_op = optimizer.apply_gradients(zip(grads, tvars))

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
