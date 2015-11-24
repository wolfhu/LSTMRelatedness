"""Module for constructing Child Sum Tree LSTM Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn import linear
from tensorflow.models.rnn.rnn_cell import RNNCell

class ChildSumTreeLSTMCell(RNNCell):
  """Child Sum Tree Long short-termmemory unit recurrent
  network cell.

  This implementation is based on:

    http://arxiv.org/pdf/1503.00075v3.pdf

  Kai Sheng Tai, Richard Socher, Christopher D. Manning
  "Improved Semantic Representations From Tree-Structured Long
  Short-Term Memory Networks." CoRR, 2015.
  """

  def __init__(self, num_units, forget_bias=1.0):
    self._num_units = num_units
    self._forget_bias = forget_bias

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return 2 * self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = tf.split(1, 2, state)
      concat = linear.linear([inputs, h], 4 * self._num_units, True)

      fs = []

      # This can be made more efficient since we're doing more than needs to be
      # done, but for now w/e
      for child_state in child_states:
          c_k, h_k = tf.split(1, 2, child_state)
          concat = linear.linear([inputs, h_k], 4 * self._num_units, True)
          i_k, j_k, f_k, o_k = tf.split(1, 4, concat)
          fs.append(f_k)


      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      # TODO: forget gate for each child, probably need to split by number
      # of child states or something
      i, j, f, o = tf.split(1, 4, concat)

      # If no children just treat it like a regular lstm
      if not fs:
        fs.append(f)

      new_c = sum(c * tf.sigmoid(fs + self._forget_bias)) + tf.sigmoid(i) * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)

    return new_h, tf.concat(1, [new_c, new_h])


class ChildSumTreeLSTM(RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, size, keep_prob = 1):
    self._children = []
    self._keep_prob = keep_prob
    self._root = ChildSumTreeLSTMCell(size, forget_bias=0.0)
    self._root = rnn_cell.DropoutWrapper(self._root, output_keep_prob=keep_prob)

  @property
  def input_size(self):
    return self._root.input_size

  @property
  def output_size(self):
    return self._root.output_size

  @property
  def state_size(self):
    return sum([cell.state_size for cell in self._children]) + self._root.state_size

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with tf.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      # Can the number of cells be variable???
      for i, cell in enumerate(self._cells):
        with tf.variable_scope("Cell%d" % i):
          cur_state = tf.slice(state, [0, cur_state_pos], [-1, cell.state_size])
          cur_state_pos += cell.state_size
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    return cur_inp, tf.concat(1, new_states)
