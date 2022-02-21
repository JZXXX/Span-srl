#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2017 Timothy Dozat
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
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import numpy as np
import tensorflow as tf

#***************************************************************
def get_sizes(t):
  """"""
  
  shape = []
  for i in six.moves.range(len(t.get_shape().as_list()[:-1])):
    shape.append(tf.shape(t)[i])
  shape.append(t.get_shape().as_list()[-1])
  return shape
  
#===============================================================
def reshape(t, shape):
  """"""
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  return tf.reshape(t, shape)

#===============================================================
def orthogonal_loss(x):
  """"""
  
  output_size = x.get_shape().as_list()[-1]
  x = tf.reshape(x, [-1, output_size])
  input_size = tf.shape(x)[0]
  I = tf.eye(output_size)
  xTx = tf.matmul(x, x, transpose_a=True)
  off_diag_xTx = xTx * (1-I)
  loss = tf.nn.l2_loss(off_diag_xTx)
  return loss

#===============================================================
def dropout(inputs, keep_prob, noise_shape=None):
  """"""
  
  if isinstance(noise_shape, (tuple, list)):
    noise_shape = tf.stack(noise_shape)
  return tf.nn.dropout(inputs, keep_prob=keep_prob, noise_shape=noise_shape)

#===============================================================
def unscaled_dropout(inputs, keep_prob, noise_shape=None):
  """"""
  
  if isinstance(noise_shape, (tuple, list)):
    noise_shape = tf.stack(noise_shape)
  return tf.nn.dropout(inputs, keep_prob=keep_prob, noise_shape=noise_shape)*keep_prob

#===============================================================
def drop_mask(shape, keep_prob):
  """"""
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  ones = tf.ones(shape)
  return dropout(ones, keep_prob)

#===============================================================
def binary_mask(shape, keep_prob):
  """"""
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  ones = tf.ones(shape)
  return unscaled_dropout(ones, keep_prob)

#===============================================================
def greater_equal(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  return tf.where(tf.greater_equal(input1, input2), ones, zeros)

#===============================================================
def greater(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  return tf.where(tf.greater(input1, input2), ones, zeros)

#===============================================================
def less_equal(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  return tf.where(tf.less_equal(input1, input2), ones, zeros)

#===============================================================
def less(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  return tf.where(tf.less(input1, input2), ones, zeros)

#===============================================================
def not_equal(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  return tf.where(tf.not_equal(input1, input2), ones, zeros)

#===============================================================
def equal(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  return tf.where(tf.equal(input1, input2), ones, zeros)

#===============================================================
def where(condition, x, y, dtype=tf.float32):
  
  ones = tf.ones_like(condition, dtype=dtype)
  return tf.where(condition, x*ones, y*ones)

#===============================================================
def ones(shape, dtype=tf.float32):
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  return tf.ones(shape, dtype=dtype)

#===============================================================
def zeros(shape, dtype=tf.float32):
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  return tf.zeros(shape, dtype=dtype)

#===============================================================
def tile(inputs, multiples):
  
  if isinstance(multiples, (tuple, list)):
    multiples = tf.stack(multiples)
  return tf.tile(inputs, multiples)

#---------------------------------------------------------------
def expand_multiply(child,parent):
  '''
  :param child: n x m x dim
  :param parent: n x m x dim
  :return: n x m x m x dim
  '''
  shape_p = get_sizes(parent)
  shape_c = get_sizes(child)
  c_tile = tf.tile(child, [1, shape_p[1], 1])
  c_tile = reshape(c_tile, [shape_c[0], shape_c[1]*shape_p[1], shape_c[2]])
  p_tile = tf.tile(parent, [1, 1, shape_c[1]])
  p_tile = reshape(p_tile, [shape_p[0], shape_p[1]*shape_c[1], shape_p[2]])
  return c_tile * p_tile

#---------------------------------------------------------------
def dl_loss(targets, probs, weights, gamma):
  '''
  :param targets:
  :param probs:
  :param weights:
  :return:
  '''
  targets = tf.to_float(targets)

  loss = 1-(2*probs*targets+gamma)/(probs**2+targets**2+gamma)
  loss = tf.reduce_sum(loss*weights)
  loss = loss/tf.reduce_sum(weights)

  return loss


#---------------------------------------------------------------
def dsc_loss(targets, probs, weights, gamma):
  '''
  :param targets:
  :param probs:
  :param weights:
  :return:
  '''
  targets = tf.to_float(targets)

  loss = 1-((1-probs)*probs*targets+gamma)/((1-probs)*probs+targets+gamma)
  loss = tf.reduce_sum(loss*weights)
  loss = loss/tf.reduce_sum(weights)

  return loss