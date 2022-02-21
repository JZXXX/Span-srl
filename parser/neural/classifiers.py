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

import numpy as np
import tensorflow as tf

from . import nn
from . import nonlin
import pdb
#***************************************************************
def hidden(layer, hidden_size, hidden_func=nonlin.relu, hidden_keep_prob=1.):
	""""""

	layer_shape = nn.get_sizes(layer)
	input_size = layer_shape.pop()
	weights = tf.get_variable('Weights', shape=[input_size, hidden_size])#, initializer=tf.orthogonal_initializer)
	biases = tf.get_variable('Biases', shape=[hidden_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		if len(layer_shape) > 1:
			noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		else:
			noise_shape = None
		layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
	
	layer = nn.reshape(layer, [-1, input_size])
	layer = tf.matmul(layer, weights) + biases
	layer = hidden_func(layer)
	layer = nn.reshape(layer, layer_shape + [hidden_size])
	return layer

#===============================================================
def hiddens(layer, hidden_sizes, hidden_func=nonlin.relu, hidden_keep_prob=1.):
	""""""
	#pdb.set_trace()
	layer_shape = nn.get_sizes(layer)
	input_size = layer_shape.pop()
	weights = []
	for i, hidden_size in enumerate(hidden_sizes):
		weights.append(tf.get_variable('Weights-%d' % i, shape=[input_size, hidden_size]))#, initializer=tf.orthogonal_initializer))
	weights = tf.concat(weights, axis=1)
	hidden_size = sum(hidden_sizes)
	biases = tf.get_variable('Biases', shape=[hidden_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		if len(layer_shape) > 1:
			noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		else:
			noise_shape = None
		layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
	
	layer = nn.reshape(layer, [-1, input_size])
	layer = tf.matmul(layer, weights) + biases
	layer = hidden_func(layer)
	layer = nn.reshape(layer, layer_shape + [hidden_size])
	layers = tf.split(layer, hidden_sizes, axis=-1)
	return layers

#===============================================================
def linear_classifier(layer, output_size, hidden_keep_prob=1.):
	""""""
	
	layer_shape = nn.get_sizes(layer)
	input_size = layer_shape.pop()
	weights = tf.get_variable('Weights', shape=[input_size, output_size], initializer=tf.zeros_initializer)
	biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		if len(layer_shape) > 1:
			noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		else:
			noise_shape = None
		layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
	
	# (n x m x d) -> (nm x d)
	layer_reshaped = nn.reshape(layer, [-1, input_size])
	
	# (nm x d) * (d x o) -> (nm x o)
	layer = tf.matmul(layer_reshaped, weights) + biases
	# (nm x o) -> (n x m x o)
	layer = nn.reshape(layer, layer_shape + [output_size])
	return layer
	
#===============================================================
def linear_attention(layer, hidden_keep_prob=1.):
	""""""
	
	layer_shape = nn.get_sizes(layer)
	input_size = layer_shape.pop()
	weights = tf.get_variable('Weights', shape=[input_size, 1], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		if len(layer_shape) > 1:
			noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		else:
			noise_shape = None
		layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
	
	# (n x m x d) -> (nm x d)
	layer_reshaped = tf.reshape(layer, [-1, input_size])
	
	# (nm x d) * (d x 1) -> (nm x 1)
	attn = tf.matmul(layer_reshaped, weights)
	# (nm x 1) -> (n x m)
	attn = tf.reshape(attn, layer_shape)
	# (n x m) -> (n x m)
	attn = tf.nn.sigmoid(attn)
	# (n x m) -> (n x 1 x m)
	soft_attn = tf.expand_dims(attn, axis=-2)
	# (n x 1 x m) * (n x m x d) -> (n x 1 x d)
	weighted_layer = tf.matmul(soft_attn, layer)
	# (n x 1 x d) -> (n x d)
	weighted_layer = tf.squeeze(weighted_layer, -2)
	return attn, weighted_layer

#===============================================================
def deep_linear_attention(layer, hidden_size, hidden_func=tf.identity, hidden_keep_prob=1.):
	""""""
	
	layer_shape = nn.get_sizes(layer)
	input_size = layer_shape.pop()
	weights = tf.get_variable('Weights', shape=[input_size, hidden_size+1], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		if len(layer_shape) > 1:
			noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		else:
			noise_shape = None
		layer = nn.dropout(layer, hidden_keep_prob, noise_shape=noise_shape)
	
	# (n x m x d) -> (nm x d)
	layer_reshaped = tf.reshape(layer, [-1, input_size])
	
	# (nm x d) * (d x o+1) -> (nm x o+1)
	attn = tf.matmul(layer_reshaped, weights)
	# (nm x o+1) -> (nm x 1), (nm x o)
	attn, layer = tf.split(attn, [1, hidden_size], axis=-1)
	# (nm x 1) -> (nm x 1)
	attn = tf.nn.sigmoid(attn)
	# (nm x 1) o (nm x o) -> (nm x o)
	weighted_layer = hidden_func(layer) * attn
	# (nm x 1) -> (n x m)
	attn = tf.reshape(attn, layer_shape)
	# (nm x o) -> (n x m x o)
	weighted_layer = nn.reshape(weighted_layer, layer_shape+[hidden_size])
	return attn, weighted_layer
	
#===============================================================
def batch_bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob, add_linear=True):
	""""""

	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+add_linear
	input2_size = layer2.get_shape().as_list()[-1]+add_linear
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-add_linear])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
	if add_linear:
		ones = tf.ones(ones_shape)
		layer1 = tf.concat([layer1, ones], -1)
		layer2 = tf.concat([layer2, ones], -1)
		biases = 0
	else:
		biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
		# (o) -> (o x 1)
		biases = nn.reshape(biases, [output_size, 1])
	
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (nm x d x 1)
	layer2 = nn.reshape(layer2, [-1, input2_size, 1])
	# (d x o x d) -> (d x od)
	weights = nn.reshape(weights, [input1_size, output_size*input2_size])
	
	# (nm x d) * (d x od) -> (nm x od)
	layer = tf.matmul(layer1, weights)
	# (nm x od) -> (nm x o x d)
	layer = nn.reshape(layer, [-1, output_size, input2_size])
	# (nm x o x d) * (nm x d x 1) -> (nm x o x 1)
	layer = tf.matmul(layer, layer2)
	# (nm x o x 1) -> (n x m x o)
	layer = nn.reshape(layer, layer_shape + [output_size]) + biases
	return layer

#===============================================================
def bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob=1., add_linear=True, target_model='CRF', tri_std=0.5):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+add_linear
	input2_size = layer2.get_shape().as_list()[-1]+add_linear
	ones_shape = tf.stack(layer_shape + [1])
	
	if target_model=='CRF':
		weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	elif target_model=='LBP':
		weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	#weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.random_normal_initializer(stddev=0.01))
	
	#weights = tf.get_variable('Weights', shape=[input1_size, output_size, input2_size], initializer=tf.contrib.layers.xavier_initializer())
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-add_linear])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
	if add_linear:#add linear means linear layer?
		ones = tf.ones(ones_shape)
		layer1 = tf.concat([layer1, ones], -1)
		layer2 = tf.concat([layer2, ones], -1)
		biases = 0
	else:
		biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
		# (o) -> (o x 1)
		biases = nn.reshape(biases, [output_size, 1])
	
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (n x m x d)
	layer2 = nn.reshape(layer2, [-1, bucket_size, input2_size])
	# (d x o x d) -> (d x od)
	weights = nn.reshape(weights, [input1_size, output_size*input2_size])
	
	# (nm x d) * (d x od) -> (nm x od)
	layer = tf.matmul(layer1, weights)
	# (nm x od) -> (n x mo x d)
	layer = nn.reshape(layer, [-1, bucket_size*output_size, input2_size])
	# (n x mo x d) * (n x m x d) -> (n x mo x m)
	layer = tf.matmul(layer, layer2, transpose_b=True)
	# (n x mo x m) -> (n x m x o x m)
	layer = nn.reshape(layer, layer_shape + [output_size, bucket_size]) + biases
	#return layer, weights
	return layer

#===============================================================
def trilinear_classifier(layer1, layer2, layer3, output_size, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#(d x d x o^2 x d) 
	#weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, output_size, input3_size], initializer=tf.truncated_normal_initializer())
	#(o^2 x d x d x d) 
	weights = tf.get_variable('trilinear_Weights', shape=[output_size**2, input1_size, input2_size, input3_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	if add_linear:
		ones = tf.ones(ones_shape)
		layer1 = tf.concat([layer1, ones], -1)
		layer2 = tf.concat([layer2, ones], -1)
		layer3 = tf.concat([layer3, ones], -1)
		biases = 0
	else:
		biases = tf.get_variable('Biases', shape=[output_size**2], initializer=tf.zeros_initializer)
		# (o) -> (o x 1)
		biases = nn.reshape(biases, [output_size**2, 1])
	#pdb.set_trace()
	# (n x m x d) -> (n x m x d)
	layer1 = nn.reshape(layer1, [-1, bucket_size, input1_size])
	# (n x m x d) -> (n x m x d) why do this?
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	# (n x m x d) -> (n x m x d)
	layer3 = tf.reshape(layer3, [-1, bucket_size, input3_size])
	
	# (n x ma x d) * (o^2 x d x d x d) -> (n x o^2 x ma x d x d)
	layer = tf.einsum('oijk,abi->aobjk', weights, layer1)
	#layer = tf.tensordot(layer1, weights, axes=[[1], [1]]) 
	# (nm x o^2 x d x d) -> (n x m x o^2 x d x d)
	#layer = nn.reshape(layer, [-1, output_size, bucket_size, input2_size, input3_size])
	# (n x o^2 x ma x d x d) * (n x mc x d) -> (n x o^2 x ma x d x mc)
	# here ab infers the n and m in layer, and a c infer n m in layer2, so abij,acj->abic
	layer = tf.einsum('aobij,acj->aobic', layer, layer3)
	# (n x o^2 x ma x d x mc) * (n x mb x d) -> (n x o^2 x ma x mb x mc)
	layer = tf.einsum('aobic,adi->aobdc', layer, layer2)
	layer += biases
	# layer = (n x o^2 x ma x mb x mc) -> (n x o x o x ma x mb x mc)
	layer = tf.reshape(layer, [-1, output_size, output_size] + [bucket_size]*3)
	#return layer
	return layer

#===============================================================
def diagonal_bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob=1., add_linear=True):
	""""""
	#here is token classifier
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input_size, output_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
	
	if add_linear: #why here do not use weights2?
		# (n x m x d) -> (nm x d)
		lin_layer1 = nn.reshape(layer1, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer1 = tf.matmul(lin_layer1, weights1)
		# (nm x o) -> (n x m x o)
		lin_layer1 = nn.reshape(lin_layer1, layer_shape + [output_size])
		# (n x m x o) -> (n x m x o x 1)
		lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer2 = tf.matmul(lin_layer2, weights1)
		# (nm x o) -> (n x m x o)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer2 = tf.transpose(lin_layer2, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=-3)
	
	# (n x m x d) -> (n x m x 1 x d)
	layer1 = nn.reshape(layer1, [-1, bucket_size, 1, input_size])
	# (n x m x d) -> (n x m x d)
	layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
	# (d x o) -> (o x d)
	weights = tf.transpose(weights, [1, 0])
	# (o) -> (o x 1)
	biases = nn.reshape(biases, [output_size, 1])
	# means every word in layer1 have m label?
	# (n x m x 1 x d) (*) (o x d) -> (n x m x o x d)
	layer = layer1 * weights
	# (n x m x o x d) -> (n x mo x d)
	layer = nn.reshape(layer, [-1, bucket_size*output_size, input_size])
	# (n x mo x d) * (n x m x d) -> (n x mo x m)
	layer = tf.matmul(layer, layer2, transpose_b=True)
	# (n x mo x m) -> (n x m x o x m)
	layer = nn.reshape(layer, layer_shape + [output_size, bucket_size])
	if add_linear:
		# (n x m x o x m) + (n x 1 x o x m) + (n x m x o x 1) -> (n x m x o x m)
		layer += lin_layer1 + lin_layer2
	# (n x m x o x m) + (o x 1) -> (n x m x o x m)
	layer += biases
	return layer

#****************************************************************
# ===============================================================
def diagonal_trilinear_classifier1(layer1, layer2, layer3, output_size, hidden_keep_prob=1., add_linear=False):
	"""
	If output**2 < d, we can use this, otherwise  diagonal_trilinear_classifier2
	:param layer1: parent
	:param layer2: child1
	:param layer3: child2
	:param output_size: len(role) * len(role)
	:param hidden_keep_prob: 
	:param add_linear: can't implement add_linear=true
	:return: 
	"""
	# here is token classifier
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	input3_size = layer3.get_shape().as_list()[-1]
	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	assert input2_size == input3_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])

	weights = tf.get_variable('Weights', shape=[output_size**2, input_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
		weights3 = tf.get_variable('Weights3', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	biases = tf.get_variable('Biases', shape=[output_size**2,1], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape)

	if add_linear:  # why here do not use weights2?
		# (n x m x d) -> (nm x d)
		lin_layer1 = nn.reshape(layer1, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer1 = tf.matmul(lin_layer1, weights1)
		# (nm x o) -> (n x m x o)
		lin_layer1 = nn.reshape(lin_layer1, layer_shape + [output_size])
		# (n x m x o) -> (n x m x o x 1)
		lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer2 = tf.matmul(lin_layer2, weights2)
		# (nm x o) -> (n x m x o)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer2 = tf.transpose(lin_layer2, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=-3)

		# (n x m x d) -> (nm x d)
		lin_layer3 = nn.reshape(layer3, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer3 = tf.matmul(lin_layer3, weights3)
		# (nm x o) -> (n x m x o)
		lin_layer3 = nn.reshape(lin_layer3, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer3 = tf.transpose(lin_layer3, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer3 = tf.expand_dims(lin_layer3, axis=-3)

	# n x m*m x d
	layer3 = nn.expand_multiply(layer3, layer1)
	layer3 = nn.reshape(layer3, [-1, bucket_size*bucket_size, 1, input_size])
	# n x m x d
	layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
	# (n x m*m x 1 x d) (*) (o x d) -> (n x m*m x o x d)
	layer3 = layer3 * weights
	# (n x m*m x o x d) -> (n x mmo x d)
	layer = nn.reshape(layer3, [-1, bucket_size*bucket_size*output_size**2, input_size])
	# (n x mmo x d) * (n x m x d) -> (n x mmo x m)
	layer = tf.matmul(layer, layer2, transpose_b=True)
	# (n x mmo x m) -> (n x m x m x o x m)
	layer = nn.reshape(layer, [-1, bucket_size, bucket_size, output_size**2, bucket_size])  # n x p x c x o x c

	if add_linear:
		# (n x m x o x m) + (n x 1 x o x m) + (n x m x o x 1) -> (n x m x o x m)
		layer += lin_layer1 + lin_layer2
	# (n x m x m x o x m) + (o x 1) -> (n x m x m x o x m)
	layer += biases
	# (n x p x c x o x c) -> (n x p x c x c x o)
	layer = tf.transpose(layer, perm=[0, 1, 2, 4, 3])  # n x p x c x c x o
	# n x p x c x c x o x o
	layer = nn.reshape(layer, [-1, bucket_size, bucket_size, bucket_size, output_size, output_size])
	return layer

# ===============================================================
def diagonal_trilinear_classifier2(layer1, layer2, layer3, output_size, hidden_keep_prob=1., add_linear=False):
	""""""
	# here is token classifier
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	input3_size = layer3.get_shape().as_list()[-1]
	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	assert input2_size == input3_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])

	weights = tf.get_variable('Weights', shape=[output_size**2, input_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
		weights3 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	biases = tf.get_variable('Biases', shape=[1,output_size**2], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape)

	if add_linear:  # why here do not use weights2?
		# (n x m x d) -> (nm x d)
		lin_layer1 = nn.reshape(layer1, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer1 = tf.matmul(lin_layer1, weights1)
		# (nm x o) -> (n x m x o)
		lin_layer1 = nn.reshape(lin_layer1, layer_shape + [output_size])
		# (n x m x o) -> (n x m x o x 1)
		lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer2 = tf.matmul(lin_layer2, weights2)
		# (nm x o) -> (n x m x o)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer2 = tf.transpose(lin_layer2, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=-3)

		# (n x m x d) -> (nm x d)
		lin_layer3 = nn.reshape(layer3, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer3 = tf.matmul(lin_layer3, weights3)
		# (nm x o) -> (n x m x o)
		lin_layer3 = nn.reshape(lin_layer3, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer3 = tf.transpose(lin_layer3, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer3 = tf.expand_dims(lin_layer3, axis=-3)

	layer = nn.expand_multiply(layer2,layer1)
	layer = nn.expand_multiply(layer3,layer)
	layer = tf.matmul(tf.reshape(layer, [-1, input_size]), weights, transpose_b=True)
	layer = nn.reshape(layer, [-1, bucket_size, bucket_size, bucket_size, output_size**2])  # n x p x c x c x o
	if add_linear:
		# (n x m x o x m) + (n x 1 x o x m) + (n x m x o x 1) -> (n x m x o x m)
		layer += lin_layer1 + lin_layer2
	# (n x m x m x m x o) + (1 x o) -> (n x m x m x m x o)
	layer += biases
	# n x p x c x c x o x o
	layer = nn.reshape(layer, [-1, bucket_size, bucket_size, bucket_size, output_size, output_size])
	return layer

# ===============================================================
def outer_trilinear_classifier(layer1, layer2, layer3, role1, role2, hidden_keep_prob=1., add_linear=False, hidde_k=200):
	"""
	layer2 and layer3; role1 and role2 are the same embedding, may (weights2 weights3), (weights4, weights5) just need one.
	:param layer1:
	:param layer2:
	:param layer3:
	:param role1:
	:param role2:
	:param hidden_keep_prob:
	:param add_linear:
	:param hidde_k:
	:return:
	"""
	# here is token classifier
	layer_shape = nn.get_sizes(layer1)
	role_shape = nn.get_sizes(role1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	input3_size = layer3.get_shape().as_list()[-1]
	nr = role1.get_shape().as_list()[0]
	role1_size  = role1.get_shape().as_list()[-1]
	role2_size = role2.get_shape().as_list()[-1]
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])

	weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights2 = tf.get_variable('trilinear_Weights2', shape=[input2_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights3 = tf.get_variable('trilinear_Weights3', shape=[input3_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights4 = tf.get_variable('trilinear_Weights4', shape=[role1_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights5 = tf.get_variable('trilinear_Weights5', shape=[role2_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	biases = tf.get_variable('Biases', shape=[nr,nr], initializer=tf.zeros_initializer)

	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
		weights3 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))

	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights4))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights5))
	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input1_size])
		# noise_shape_role = tf.stack(role_shape[:-1] + [1, input1_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape)
		# role1 = nn.dropout(role1, hidden_keep_prob, noise_shape=noise_shape_role)
		# role2 = nn.dropout(role2, hidden_keep_prob, noise_shape=noise_shape_role)

	if add_linear:  # why here do not use weights2?
		# (n x m x d) -> (nm x d)
		lin_layer1 = nn.reshape(layer1, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer1 = tf.matmul(lin_layer1, weights1)
		# (nm x o) -> (n x m x o)
		lin_layer1 = nn.reshape(lin_layer1, layer_shape + [output_size])
		# (n x m x o) -> (n x m x o x 1)
		lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer2 = tf.matmul(lin_layer2, weights2)
		# (nm x o) -> (n x m x o)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer2 = tf.transpose(lin_layer2, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=-3)

		# (n x m x d) -> (nm x d)
		lin_layer3 = nn.reshape(layer3, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer3 = tf.matmul(lin_layer3, weights3)
		# (nm x o) -> (n x m x o)
		lin_layer3 = nn.reshape(lin_layer3, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer3 = tf.transpose(lin_layer3, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer3 = tf.expand_dims(lin_layer3, axis=-3)

	# (n x mp x d) * (d x k) -> (n x mp x k)
	layer1 = tf.tensordot(layer1, weights1)
	# (n x mc1 x d) * (d x k) -> (n x mc1 x k)
	layer2 = tf.tensordot(layer2, weights2)
	# (n x mc2 x d) * (d x k) -> (n x mc2 x k)
	layer3 = tf.tensordot(layer3, weights3)
	# (nr x d) * (d x k) -> (nr x k)
	role1 = tf.tensordot(role1, weights4)
	# (nr x d) * (d x k) -> (nr x k)
	role2 = tf.tensordot(role2, weights5)

	# (n x mc1 x k), (nr x k) -> (n x mc1 x nr x k)
	layer2_role1 = tf.einsum('nak,rk->nark',layer2,role1)
	# (n x mc2 x k), (nr x k) -> (n x mc2 x nr x k)
	layer3_role2 = tf.einsum('nbk,rk->nbrk',layer3,role2)
	# (n x mp x k), (n x mc1 x nr x k) -> (n x mp x mc1 x nr x k)
	layer1_layer2 = tf.einsum('npk,nark->npark', layer1, layer2_role1)
	# (n x mp x mc1 x nr x k), (n x mc2 x nr x k) -> (n x mp x mc1 x mc2 x nr x nr)
	layer = tf.einsum('npark,mbok->npabro', layer1_layer2, layer3_role2)

	if add_linear:
		# (n x m x o x m) + (n x 1 x o x m) + (n x m x o x 1) -> (n x m x o x m)
		layer += lin_layer1 + lin_layer2
	# (n x m x m x m x nr x nr) + (nr x nr) -> (n x m x m x m x nr x nr)
	layer += biases
	return layer

# ===============================================================
def cross_trilinear_classifier(layer1, layer2, layer3, role1, role2, hidden_keep_prob=1., add_linear=False, hidde_k = 200):
	"""
	:param layer1: parent embedding n x mp x d
	:param layer2: child1 embedding n x mc1 x d
	:param layer3: child2 embedding n x mc2 x d
	:param role1: selected roles embedding 10 x d
	:param role2: masked n x mp x mc2 x d
	:param hidden_keep_prob:
	:param add_linear:
	:param hidde_k:
	:return:
	"""
	# here is token classifier
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	input3_size = layer3.get_shape().as_list()[-1]
	role_size  = role1.get_shape().as_list()[-1]
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])

	weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights2 = tf.get_variable('trilinear_Weights2', shape=[input2_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights3 = tf.get_variable('trilinear_Weights3', shape=[input3_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights4 = tf.get_variable('trilinear_Weights4', shape=[role_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights5 = tf.get_variable('trilinear_Weights5', shape=[role_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))

	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
		weights3 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))

	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights4))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights5))

	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input1_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape)

	if add_linear:  # why here do not use weights2?
		# (n x m x d) -> (nm x d)
		lin_layer1 = nn.reshape(layer1, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer1 = tf.matmul(lin_layer1, weights1)
		# (nm x o) -> (n x m x o)
		lin_layer1 = nn.reshape(lin_layer1, layer_shape + [output_size])
		# (n x m x o) -> (n x m x o x 1)
		lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer2 = tf.matmul(lin_layer2, weights2)
		# (nm x o) -> (n x m x o)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer2 = tf.transpose(lin_layer2, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=-3)

		# (n x m x d) -> (nm x d)
		lin_layer3 = nn.reshape(layer3, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer3 = tf.matmul(lin_layer3, weights3)
		# (nm x o) -> (n x m x o)
		lin_layer3 = nn.reshape(lin_layer3, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer3 = tf.transpose(lin_layer3, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer3 = tf.expand_dims(lin_layer3, axis=-3)

	# (n x mp x d) * (d x k) -> (n x mp x k)
	layer1 = tf.tensordot(layer1, weights1)
	# (n x mc1 x d) * (d x k) -> (n x mc1 x k)
	layer2 = tf.tensordot(layer2, weights2)
	# (n x mc2 x d) * (d x k) -> (n x mc2 x k)
	layer3 = tf.tensordot(layer3, weights3)
	# (10 x d) * (d x k) -> (10 x k)
	role1 = tf.tensordot(role1, weights4)
	# (n x mp x mc2 x d) * (d x k) -> (n x mp x mc2 x k)
	role2 = tf.tensordot(role2, weights5)

	# (n x mp x k), (n x mc1 x k) -> (n x mp x mc1 x k)
	layerpc1 = tf.einsum('npk,nak->npak', layer1, layer2)
	# (n x mp x k), (n x mc2 x k) -> (n x mp x mc2 x k)
	layerpc2 = tf.einsum('npk,nak->npak', layer1, layer3)
	# (n x mp x mc2 x k), (n x mp x mc2 x k) -> (n x mp x mc2 x k)
	layer3_role2 = layerpc2 * role2
	# (n x mp x mc1 x k), (10 x k) -> (n x mp x mc1 x 10 x k)
	layer2_role1 = tf.einsum('npak,rk->npark', layerpc1, role1)
	# (n x mp x mc1 x 10 x k), (n x mp x mc2 x k) -> (n x mp x mc1 x mc2 x 10)
	layer_temp = tf.einsum('npark,npbk->npabr', layer2_role1,layer3_role2)
	# (n x mp x mc1 x mc2 x 10) -> (n x mp x mc1 x 10)
	layer = tf.reduce_sum(layer_temp, axis=-2)

	if add_linear:
		# (n x m x o x m) + (n x 1 x o x m) + (n x m x o x 1) -> (n x m x o x m)
		layer += lin_layer1 + lin_layer2
	# (n x m x m x m x nr x nr) + (nr x nr) -> (n x m x m x m x nr x nr)
	return layer

#===============================================================
def diagonal_bilinear_layer(layer1, layer2, hidden_func=nonlin.relu, hidden_keep_prob=1., add_linear=True):
	""""""
	#here is token classifier
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input_size, input_size], initializer=tf.random_normal_initializer(stddev=0.1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size, input_size], initializer=tf.random_normal_initializer(stddev=0.1))
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size, input_size], initializer=tf.random_normal_initializer(stddev=0.1))
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	biases = tf.get_variable('Biases', shape=[input_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
	
	if add_linear:
		# (n x m x d) * (d x o) -> (n x m x o) -> # (n x m x o) -> (n x m x 1 x o)
		lin_layer1 = tf.expand_dims(tf.tensordot(layer1, weights1, axes=[[-1],[0]]),axis=-2)
		# (n x m x d) * (d x o) -> (n x m x o) -> # (n x m x o) -> (n x 1 x m x o)
		lin_layer2 = tf.expand_dims(tf.tensordot(layer2, weights2, axes=[[-1],[0]]),axis=-3)
	# (n x m x d) -> (n x m x d x 1)
	layer1 = tf.expand_dims(layer1,axis=-1)
	# (n x m x d x 1) * (d x o) -> (n x m x d x o)
	layer = layer1 * weights
	# (n x m x d x o) * (n x m x d) -> (n x m x m x o)
	layer = tf.einsum('nmdo,nad->nmao', layer, layer2)
	if add_linear:
		# (n x m x m x o) + (n x m x 1 x o) + (n x 1 x m x o) -> (n x m x m x o)
		layer += lin_layer1 + lin_layer2
	# (o) -> (1 x o)
	biases = nn.reshape(biases, [1,input_size])
	# (n x m x m x o) + (1 x o) -> (n x m x m x o)
	layer += biases
	# (n x m x m x d)
	#layer = hidden_func(layer)
	return layer

#===============================================================
def trilinear_label_layer(label_layer, hidden_keep_prob=1., weight_type=1):
	""""""
	#here is token classifier
	layer_shape = nn.get_sizes(label_layer)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])
	if weight_type==1:
		weights = tf.get_variable('Weights', shape=[input_size, input_size], initializer=tf.random_normal_initializer(stddev=0.01))
		weights2 = tf.get_variable('Weights2', shape=[input_size, input_size], initializer=tf.random_normal_initializer(stddev=0.01))
		weights3 = tf.get_variable('Weights3', shape=[input_size, input_size], initializer=tf.random_normal_initializer(stddev=0.01))
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
		biases1 = tf.get_variable('Biases1', shape=[1, input_size], initializer=tf.zeros_initializer)
		biases2 = tf.get_variable('Biases2', shape=[1, input_size], initializer=tf.zeros_initializer)
		biases3 = tf.get_variable('Biases3', shape=[1, input_size], initializer=tf.zeros_initializer)
		#======================way1================================
		#(n x m x m x d) * (d x d) -> (n x m x m x d)
		label_layer1=tf.tensordot(label_layer,weights, axes=[[-1],[0]])+biases1
		label_layer2=tf.tensordot(label_layer,weights2, axes=[[-1],[0]])+biases2
		label_layer3=tf.tensordot(label_layer,weights3, axes=[[-1],[0]])+biases3
	if weight_type==2:
		weights = tf.get_variable('Weights', shape=[1,input_size], initializer=tf.truncated_normal_initializer())
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
		#======================way1================================
		#(n x m x m x d) * (d x d) -> (n x m x m x d)
		label_layer=label_layer*weights
	#(n x ma x mb x d) * (n x ma x mc x d) -> (n x ma x mb x mc)
	label_sib=tf.einsum('nabd,nacd->nabc',label_layer1, label_layer1)
	label_cop=tf.einsum('nabd,ncbd->nabc',label_layer2, label_layer2)
	label_gp=tf.einsum('nabd,nbcd->nabc',label_layer3, label_layer3)
	return label_sib,label_cop,label_gp

#===============================================================
def bilinear_discriminator(layer1, layer2, hidden_keep_prob=1., add_linear=True, target_model='CRF', tri_std=0.5):
	""""""
	#pdb.set_trace()
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	if target_model=='CRF':
		weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	elif target_model=='LBP':
		weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	else:
		weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.zeros_initializer)
	#weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (n x m x d)
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	
	# (nm x d) * (d x d) -> (nm x d)
	layer = tf.matmul(layer1, weights) #here is the u matrix in the paper?
	# (nm x d) -> (n x m x d)
	layer = nn.reshape(layer, [-1, bucket_size, input2_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	layer = tf.matmul(layer, layer2, transpose_b=True)
	# (n x mo x m) -> (n x m x m)
	layer = nn.reshape(layer, layer_shape + [bucket_size])
	# (n x ma x mb)
	return layer

#===============================================================
def dist_bilinear_discriminator(layer1, layer2, hidden_keep_prob=1., add_linear=True, target_model='CRF', tri_std=0.5):
	""""""
	# (n x ma x d1) * (n x ma x mb x d2) -> (n x ma x mb)
	#pdb.set_trace()
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1

	layer2_shape = nn.get_sizes(layer2)
	bucket_size2 = layer2_shape[-2]
	input2_size = layer2_shape.pop()+1
	ones1_shape = tf.stack(layer_shape + [1])
	ones2_shape = tf.stack(layer2_shape + [1])
	if target_model=='CRF':
		weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	elif target_model=='LBP':
		weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	#weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer2_shape[:-1] + [1, input2_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
	ones1 = tf.ones(ones1_shape)
	ones2 = tf.ones(ones2_shape)
	layer1 = tf.concat([layer1, ones1], -1)
	layer2 = tf.concat([layer2, ones2], -1)
	
	# (n x ma x d1) -> (nma x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x ma x mb x d) -> (nma x mb x d)
	layer2 = tf.reshape(layer2, [-1, bucket_size2, input2_size])
	
	# (nma x d1) * (d1 x d2) -> (nma x d2)
	layer = tf.matmul(layer1, weights)
	# # (nma x d2) -> (n x ma x d2)
	# layer = nn.reshape(layer, [-1, bucket_size, input2_size])
	# (nma x d2) * (nma x mb x d2) -> (nma x mb)
	layer = tf.einsum('nd,nbd->nb', layer, layer2)
	# (nma x mb) -> (n x ma x mb)
	layer = nn.reshape(layer, [-1, bucket_size, bucket_size2])

	return layer

#===============================================================
def simple_bilinear_discriminator(layer1, layer2, hidden_keep_prob=1., add_linear=True, target_model='CRF', tri_std=0.5):
	""""""
	# (nmamb x d1) * (nmamb x d2) -> (n x ma x mb)
	#pdb.set_trace()
	layer_shape = nn.get_sizes(layer1)
	input1_size = layer_shape.pop()+1

	layer2_shape = nn.get_sizes(layer2)
	input2_size = layer2_shape.pop()+1
	ones_shape = tf.stack(layer_shape + [1])
	if target_model=='CRF':
		weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	elif target_model=='LBP':
		weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	#weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer2_shape[:-1] + [1, input2_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	
	# (nmamb x d1) * (d1 x d2) -> (nmamb x d2)
	layer = tf.matmul(layer1, weights)
	# (nmamb x d2) * (nmamb x d2) -> (nmamb)
	layer = tf.einsum('nd,nd->n',layer, layer2)
	
	return layer

#===============================================================
def trilinear_discriminator_old(layer1, layer2, layer3, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#(d x d x d) layer1=axis0, layer2=axis2, layer3=axis1
	weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input3_size, input2_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	layer3 = tf.concat([layer3, ones], -1)
	#pdb.set_trace()
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (nm x d) why do this?
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	# (n x m x d) -> (nm x d)
	layer3 = tf.reshape(layer3, [-1, bucket_size, input3_size])
	
	# (nm x d) * (d x d x d) -> (nm x d x d)
	layer = tf.tensordot(layer1, weights, axes=[[1], [0]]) 
	# (nm x d x d) -> (nm x d x d)
	layer = nn.reshape(layer, [-1, bucket_size, input3_size, input2_size])
	# (n x m x d x d) * (n x m x d) -> (n x m x d x m)
	# here ab infers the n and m in layer, and a c infer n m in layer2, so abij,acj->abic
	layer = tf.einsum('abij,acj->abic', layer, layer2)
	# (n x m x d x m) * (n x m x d) -> (n x m x m x m)
	layer = tf.einsum('abic,adi->abdc', layer, layer3)
	# (n x mo x m x m) -> (n x m x m x m)
	layer = nn.reshape(layer, layer_shape + [bucket_size]*2)
	# Here return a (n x m x m x m) graph, so how to use it?
	#return layer, weights
	# layer = (n x ma x mc x mb) -> (n x ma x mb x mc)
	layer = tf.transpose(layer, perm=[0,1,3,2])
	#return layer
	return layer, weights

#===============================================================
def trilinear_discriminator_test2(layer1, layer2, layer3, hidden_keep_prob=1., add_linear=True):
	""""""
	#pdb.set_trace()
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#(d x d x d) layer1=axis0, layer2=axis2, layer3=axis1
	weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, input2_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
	weights2 = tf.get_variable('trilinear_Weights2', shape=[input2_size, input3_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	weights3 = tf.get_variable('trilinear_Weights3', shape=[input3_size, input1_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	layer3 = tf.concat([layer3, ones], -1)


	#pdb.set_trace()
	# (n x m x d) -> (nm x d)
	layer1_h = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (nm x d) why do this?
	layer2_h = tf.reshape(layer2, [-1, input2_size])
	# (n x m x d) -> (nm x d)
	layer3_h = tf.reshape(layer3, [-1, input3_size])
	


	# (nm x d) * (d x d) -> (nm x d)
	bi_layer1 = tf.matmul(layer1_h, weights1) 
	# (nm x d) -> (n x m x d)
	bi_layer1 = nn.reshape(bi_layer1, [-1, bucket_size, input1_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	bi_layer1 = tf.matmul(bi_layer1, layer2, transpose_b=True)
	# (n x m x m) -> (n x ma x mb)
	bi_layer1 = nn.reshape(bi_layer1, layer_shape + [bucket_size])
	# (n x m x m) -> (n x ma x mb x 1)
	bi_layer1 = tf.expand_dims(bi_layer1, -1)

	# (nm x d) * (d x d) -> (nm x d)
	bi_layer2 = tf.matmul(layer2_h, weights2) 
	# (nm x d) -> (n x m x d)
	bi_layer2 = nn.reshape(bi_layer2, [-1, bucket_size, input2_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	bi_layer2 = tf.matmul(bi_layer2, layer3, transpose_b=True)
	# (n x m x m) -> (n x mb x mc)
	bi_layer2 = nn.reshape(bi_layer2, layer_shape + [bucket_size])
	# (n x m x m) -> (n x 1 x mb x mc)
	bi_layer2 = tf.expand_dims(bi_layer2, -3)

	# (nm x d) * (d x d) -> (nm x d)
	bi_layer3 = tf.matmul(layer1_h, weights3) 
	# (nm x d) -> (n x m x d)
	bi_layer3 = nn.reshape(bi_layer3, [-1, bucket_size, input2_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	bi_layer3 = tf.matmul(bi_layer3, layer3, transpose_b=True)
	# (n x m x m) -> (n x ma x mc)
	bi_layer3 = nn.reshape(bi_layer3, layer_shape + [bucket_size])
	# (n x m x m) -> (n x ma x 1 x mc)
	bi_layer3 = tf.expand_dims(bi_layer3, -2)
	# (n x ma x mb x 1) + (n x 1 x mb x mc) + (n x ma x 1 x mc) -> (n x ma x mb x mc)
	layer = bi_layer1 + bi_layer2 + bi_layer3
	#return layer
	return layer

#===============================================================
def trilinear_discriminator_test(layer1, layer2, layer3, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#(d x d x d) layer1=axis0, layer2=axis2, layer3=axis1
	weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, input1_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
	weights2 = tf.get_variable('trilinear_Weights2', shape=[input2_size, input2_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	weights3 = tf.get_variable('trilinear_Weights3', shape=[input3_size, input3_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	layer3 = tf.concat([layer3, ones], -1)

	#pdb.set_trace()
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (nm x d) why do this?
	layer2 = tf.reshape(layer2, [-1, input2_size])
	# (n x m x d) -> (nm x d)
	layer3 = tf.reshape(layer3, [-1, input3_size])
	
	#pdb.set_trace()
	#Does here need any dropout?
	# (nm x d) + (nm x d) -> (nm x 2d)
	cat_layer1 = tf.concat([layer2, layer3], -1)
	# (nm x 2d) -> (nm x d)
	with tf.variable_scope('linear1'):
		cat_layer1 = hidden(cat_layer1, input1_size, hidden_func=nonlin.identity, hidden_keep_prob=1.)
	# (nm x d) + (nm x d) -> (nm x 2d)
	cat_layer2 = tf.concat([layer1, layer3], -1)
	# (nm x 2d) -> (nm x d)
	with tf.variable_scope('linear2'):
		cat_layer2 = hidden(cat_layer2, input2_size, hidden_func=nonlin.identity, hidden_keep_prob=1.)
	# (nm x d) + (nm x d) -> (nm x 2d)
	cat_layer3 = tf.concat([layer1, layer2], -1)
	# (nm x 2d) -> (nm x d)
	with tf.variable_scope('linear3'):
		cat_layer3 = hidden(cat_layer3, input3_size, hidden_func=nonlin.identity, hidden_keep_prob=1.)

	# (nm x d) -> (n x m x d)
	cat_layer1 = nn.reshape(cat_layer1, [-1, bucket_size, input2_size])
	cat_layer2 = nn.reshape(cat_layer2, [-1, bucket_size, input2_size])
	cat_layer3 = nn.reshape(cat_layer3, [-1, bucket_size, input2_size])

	# (nm x d) * (d x d) -> (nm x d)
	bi_layer1 = tf.matmul(layer1, weights1) 
	# (nm x d) -> (n x m x d)
	bi_layer1 = nn.reshape(bi_layer1, [-1, bucket_size, input2_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	bi_layer1 = tf.matmul(bi_layer1, cat_layer1, transpose_b=True)
	# (n x m x m) -> (n x ma x mbc)
	bi_layer1 = nn.reshape(bi_layer1, layer_shape + [bucket_size])
	# (n x m x m) -> (n x ma x mbc x 1)
	bi_layer1 = tf.expand_dims(bi_layer1, -1)
	# (n x m x m) -> (n x ma x mb x mc)
	bi_layer1 = tf.tile(bi_layer1, [1,1,1,bucket_size])

	# (nm x d) * (d x d) -> (nm x d)
	bi_layer2 = tf.matmul(layer2, weights2) 
	# (nm x d) -> (n x m x d)
	bi_layer2 = nn.reshape(bi_layer2, [-1, bucket_size, input2_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	bi_layer2 = tf.matmul(bi_layer2, cat_layer2, transpose_b=True)
	# (n x m x m) -> (n x mb x mac)
	bi_layer2 = nn.reshape(bi_layer2, layer_shape + [bucket_size])
	# (n x m x m) -> (n x mb x mac x 1)
	bi_layer2 = tf.expand_dims(bi_layer2, -1)
	# (n x m x m) -> (n x mb x ma x mc)
	bi_layer2 = tf.tile(bi_layer2, [1,1,1,bucket_size])
	# (n x mb x ma x mc) -> (n x ma x mb x mc)
	bi_layer2 = tf.transpose(bi_layer2, [0,2,1,3])

	# (nm x d) * (d x d) -> (nm x d)
	bi_layer3 = tf.matmul(layer3, weights3) 
	# (nm x d) -> (n x m x d)
	bi_layer3 = nn.reshape(bi_layer3, [-1, bucket_size, input2_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	bi_layer3 = tf.matmul(bi_layer3, cat_layer3, transpose_b=True)
	# (n x m x m) -> (n x mc x mab)
	bi_layer3 = nn.reshape(bi_layer3, layer_shape + [bucket_size])
	# (n x m x m) -> (n x mc x mab x 1)
	bi_layer3 = tf.expand_dims(bi_layer3, -1)
	# (n x m x m) -> (n x mc x ma x mb)
	bi_layer3 = tf.tile(bi_layer3, [1,1,1,bucket_size])
	# (n x mc x ma x mb) -> (n x ma x mb x mc)
	bi_layer3 = tf.transpose(bi_layer3, [0,2,3,1])

	layer = bi_layer1 + bi_layer2 + bi_layer3
	#return layer
	return layer

#===============================================================
def trilinear_discriminator_new(layer1, layer2, layer3, hidden_keep_prob=1., add_linear=True,target_model='CRF',tri_std=0.01):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#(d x d x d) layer1=axis0, layer2=axis2, layer3=axis1
	if target_model=='CRF':
		#weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, input3_size], initializer=tf.truncated_normal_initializer())
		weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, input3_size], initializer=tf.random_normal_initializer(stddev=tri_std))
		
	elif target_model=='LBP':
		weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, input3_size], initializer=tf.random_normal_initializer(stddev=tri_std))
	#weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, input3_size], initializer=tf.truncated_normal_initializer())
	#weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input2_size, input3_size], initializer=tf.contrib.layers.xavier_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	layer3 = tf.concat([layer3, ones], -1)
	#pdb.set_trace()
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (n x m x d) why do this?
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	# (n x m x d) -> (n x m x d)
	layer3 = tf.reshape(layer3, [-1, bucket_size, input3_size])
	
	# (nm x d) * (d x d x d) -> (nm x d x d)
	layer = tf.tensordot(layer1, weights, axes=[[1], [0]]) 
	# (nm x d x d) -> (n x m x d x d)
	layer = nn.reshape(layer, [-1, bucket_size, input2_size, input3_size])
	# (n x m x d x d) * (n x m x d) -> (n x m x d x m)
	# here ab infers the n and m in layer, and a c infer n m in layer2, so abij,acj->abic
	layer = tf.einsum('abij,acj->abic', layer, layer3)
	# (n x m x d x m) * (n x m x d) -> (n x m x m x m)
	layer = tf.einsum('abic,adi->abdc', layer, layer2)
	# (n x mo x m x m) -> (n x m x m x m)
	layer = nn.reshape(layer, layer_shape + [bucket_size]*2)
	# layer = (n x ma x mc x mb) 
	#return layer,weights
	return layer

#===============================================================
def trilinear_discriminator_outer(layer1, layer2, layer3, hidden_keep_prob=1., add_linear=True,target_model='CRF',tri_std=0.01, hidden_k=200):
	""""""
	#use outer tensor product for trilinear layer size of k x d x 1
	#pdb.set_trace()
	#print(tri_std)
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#(d x d x d) layer1=axis0, layer2=axis2, layer3=axis1
	if target_model=='CRF':
		weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
		weights2 = tf.get_variable('trilinear_Weights2', shape=[input2_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
		weights3 = tf.get_variable('trilinear_Weights3', shape=[input3_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
	elif target_model=='LBP':
		weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
		weights2 = tf.get_variable('trilinear_Weights2', shape=[input2_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
		weights3 = tf.get_variable('trilinear_Weights3', shape=[input3_size, hidden_k], initializer=tf.random_normal_initializer(stddev=tri_std))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	layer3 = tf.concat([layer3, ones], -1)
	#pdb.set_trace()
	# (n x m x d) * (d x k) -> (n x m x k)
	layer1_tmp = tf.tensordot(layer1, weights1, axes=[[-1], [0]])
	# (n x m x d) * (d x k) -> (n x m x k)
	layer2_tmp = tf.tensordot(layer2, weights2, axes=[[-1], [0]])
	# (n x m x d) * (d x k) -> (n x m x k)
	layer3_tmp = tf.tensordot(layer3, weights3, axes=[[-1], [0]])
	#(n x ma x k) * (n x mb x k) -> (n x ma x mb x k)
	layer12_tmp = tf.einsum('nak,nbk->nabk',layer1_tmp,layer2_tmp)
	#(n x ma x mb x k) * (n x mc x k) -> (n x ma x mb x mc)
	layer = tf.einsum('nabk,nck->nabc',layer12_tmp,layer3_tmp)
	#layer=tf.reduce_sum(layer123,1)
	return layer

#===============================================================
def trilinear_discriminator(layer1, layer2, layer3, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#(d x d x d) layer1=axis0, layer2=axis2, layer3=axis1
	weights = tf.get_variable('trilinear_sibling_Weights', shape=[input1_size, input3_size, input2_size], initializer=tf.truncated_normal_initializer())
	weights2 = tf.get_variable('trilinear_grandparent_Weights', shape=[input3_size, input1_size, input2_size], initializer=tf.truncated_normal_initializer())
	with tf.device('/device:GPU:2'):  
		weights3 = tf.get_variable('trilinear_coparent_Weights', shape=[input2_size, input3_size, input1_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	layer3 = tf.concat([layer3, ones], -1)
	#pdb.set_trace()
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (n x m x d) why do this?
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	# (n x m x d) -> (n x m x d)
	layer3 = tf.reshape(layer3, [-1, bucket_size, input3_size])
	#=============================================================
	# First part, sibling potential
	# (ha x hc x hb)
	# (nm x d) * (d x d x d) -> (nm x d x d)
	layer_sib = tf.tensordot(layer1, weights, axes=[[1], [0]]) 
	# (nm x d x d) -> (nm x d x d)
	layer_sib = nn.reshape(layer_sib, [-1, bucket_size, input3_size, input2_size])
	# (n x m x d x d) * (n x m x d) -> (n x m x d x m)
	# here ab infers the n and m in layer, and a c infer n m in layer2, so abij,acj->abic
	layer_sib = tf.einsum('abij,acj->abic', layer_sib, layer2)
	# (n x m x d x m) * (n x m x d) -> (n x m x m x m)
	layer_sib = tf.einsum('abic,adi->abdc', layer_sib, layer3)
	# (n x mo x m x m) -> (n x m x m x m)
	layer_sib = nn.reshape(layer_sib, layer_shape + [bucket_size]*2)

	#=============================================================
	# Second part, grandparent potential
	# (hc x ha x hb)

	# (n x m x d) -> (nm x d)
	layer3 = nn.reshape(layer3, [-1, input3_size])
	# (n x m x d) -> (nm x d) why do this?
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	# (n x m x d) -> (nm x d)
	layer1 = tf.reshape(layer1, [-1, bucket_size, input1_size])

	# (nm x d) * (d x d x d) -> (nm x d x d)
	layer_gp = tf.tensordot(layer3, weights2, axes=[[1], [0]]) 
	# (nm x d x d) -> (nm x d x d)
	layer_gp = nn.reshape(layer_gp, [-1, bucket_size, input1_size, input2_size])
	# (n x m x d x d) * (n x m x d) -> (n x m x d x m)
	# here ab infers the n and m in layer, and a c infer n m in layer2, so abij,acj->abic
	layer_gp = tf.einsum('abij,acj->abic', layer_gp, layer2)
	# (n x m x d x m) * (n x m x d) -> (n x m x m x m)
	layer_gp = tf.einsum('abic,adi->abdc', layer_gp, layer1)
	# (n x mo x m x m) -> (n x m x m x m)
	layer_gp = nn.reshape(layer_gp, layer_shape + [bucket_size]*2)

	#=============================================================
	# Third part, coparent potential
	# (hb x hc x ha)
	with tf.device('/device:GPU:2'):  
		# (n x m x d) -> (nm x d)
		layer2 = nn.reshape(layer2, [-1, input2_size])
		# (n x m x d) -> (nm x d) why do this?
		layer3 = tf.reshape(layer3, [-1, bucket_size, input3_size])
		# (n x m x d) -> (nm x d)
		layer1 = tf.reshape(layer1, [-1, bucket_size, input1_size])

		# (nm x d) * (d x d x d) -> (nm x d x d)
		layer_cop = tf.tensordot(layer2, weights3, axes=[[1], [0]]) 
		# (nm x d x d) -> (nm x d x d)
		layer_cop = nn.reshape(layer_cop, [-1, bucket_size, input3_size, input1_size])
		# (n x m x d x d) * (n x m x d) -> (n x m x d x m)
		# here ab infers the n and m in layer, and a c infer n m in layer2, so abij,acj->abic
		layer_cop = tf.einsum('abij,acj->abic', layer_cop, layer1)
		# (n x m x d x m) * (n x m x d) -> (n x m x m x m)
		layer_cop = tf.einsum('abic,adi->abdc', layer_cop, layer3)
		# (n x mo x m x m) -> (n x m x m x m)
		layer_cop = nn.reshape(layer_cop, layer_shape + [bucket_size]*2)
	# wait! here is a problem, the tensor is different on each dimension! must adjust the dimension? or return three tensors to the outside?
	# TODO:
	# 1 dimension of each matrix
	# 2 parallelization in this part!
	# 3 for 1, here is a huge wrong(need transformation), then the previous simulation may have some problem!
	# layer_sib = (n x ma x mc x mb) -> (n x ma x mb x mc)
	layer_sib = tf.transpose(layer_sib, perm=[0,1,3,2])
	# layer_gp = (n x mc x ma x mb) -> (n x ma x mb x mc)
	layer_gp = tf.transpose(layer_gp, perm=[0,2,3,1])
	with tf.device('/device:GPU:2'):  
		# layer_cop = (n x mb x mc x ma) -> (n x ma x mb x mc)
		layer_cop = tf.transpose(layer_cop, perm=[0,3,1,2])
	# (x) here cannot add them up! there a something different here!
	#layer=layer_cop+layer_sib+layer_gp

	#return layer, weights
	#return layer
	return layer_sib, layer_gp, layer_cop

#===============================================================
def trilinear_discriminator2(layer1, layer2, layer3, hidden_keep_prob=1., add_linear=True):
	""""""
	# TODO: felt strange here...
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+1
	input2_size = layer2.get_shape().as_list()[-1]+1
	#here add a third layer
	input3_size = layer3.get_shape().as_list()[-1]+1
	ones_shape = tf.stack(layer_shape + [1])
	#pdb.set_trace()
	#(d x d x d) layer1=axis0, layer2=axis2, layer3=axis1
	#weights = tf.get_variable('trilinear_Weights', shape=[input1_size, input3_size, input2_size], initializer=tf.truncated_normal_initializer())
	#(d x d) for layer1 and layer2 (d x d)->(m x m)
	weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, input2_size], initializer=tf.truncated_normal_initializer())
	#(d x d) for layer1 and layer3 (d x d)->(m x m)
	weights2 = tf.get_variable('trilinear_Weights2', shape=[input1_size, input3_size], initializer=tf.truncated_normal_initializer())
	#(m x m x m) for  (m x m) * (m x m x m) * (m x m) ->(m x m x m)
	weights3 = tf.get_variable('trilinear_Weights3', shape=[bucket_size, bucket_size, bucket_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size-1])
		noise_shape3 = tf.stack(layer_shape[:-1] + [1, input3_size-1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	layer3 = tf.concat([layer3, ones], -1)
	pdb.set_trace()
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (nm x d) why do this?
	layer2 = tf.reshape(layer2, [-1, bucket_size, input2_size])
	# (n x m x d) -> (nm x d)
	layer3 = tf.reshape(layer3, [-1, bucket_size, input3_size])
	
	# (nm x d) * (d x d) -> (nm x d)
	layer_temp1 = tf.matmul(layer1, weights1)
	layer_temp2 = tf.matmul(layer1, weights2)
	# (nm x d) -> (n x m x d)
	layer_temp1 = nn.reshape(layer_temp1, [-1, bucket_size, input2_size])
	layer_temp2 = nn.reshape(layer_temp2, [-1, bucket_size, input3_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	layer_temp1 = tf.matmul(layer_temp1, layer2, transpose_b=True)
	layer_temp2 = tf.matmul(layer_temp2, layer3, transpose_b=True)

	# (n x m x m) * (m x m x m) -> (n x m x m x m)
	# (nm x m) * (m x m^2) -> (nm x m^2)
	#layer_temp1 = nn.reshape(layer_temp1, [-1, bucket_size])
	#weights3_temp = nn.reshape(weights3, [-1, bucket_size*bucket_size])
	layer = tf.tensordot(layer_temp1, weights3, axes=[[-1], [0]])
	# (n x m x m x [m]) * (n x [m] x m) -> (n x m x m x m)
	layer = tf.einsum('nabc,ncd->nabd', layer, layer2)
	#layer = tf.tensordot(layer, layer_temp2, axes=[[-1],[]])
	layer = nn.reshape(layer, layer_shape + [bucket_size]*2)
	return layer

#===============================================================
def diagonal_bilinear_discriminator(layer1, layer2, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	biases = tf.get_variable('Biases', shape=[1], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
	
	if add_linear:
		#(d) -> (d x 1)
		weights1 = tf.expand_dims(weights1, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_layer1 = nn.reshape(layer1, [-1, input_size])
		# (nm x d) * (d x 1) -> (nm x 1)
		lin_layer1 = tf.matmul(lin_layer1, weights1)
		# (nm x 1) -> (n x m)
		lin_layer1 = nn.reshape(lin_layer1, layer_shape)
		# (n x m) -> (n x m x 1)
		lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
		#(d) -> (d x 1)
		weights2 = tf.expand_dims(weights2, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x 1) -> (nm x 1)
		lin_layer2 = tf.matmul(lin_layer2, weights1)
		# (nm x 1) -> (n x m)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape)
		# (n x m) -> (n x 1 x m)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=-2)
	
	# (n x m x d) -> (n x m x d)
	layer1 = nn.reshape(layer1, [-1, bucket_size, input_size])
	# (n x m x d) -> (n x m x d)
	layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
	
	# (n x m x d) (*) (d) -> (n x m x d)
	layer = layer1 * weights
	# (n x m x d) * (n x m x d) -> (n x m x m)
	layer = tf.matmul(layer, layer2, transpose_b=True)
	# (n x m x m) -> (n x m x m)
	layer = nn.reshape(layer, layer_shape + [bucket_size])
	if add_linear:
		# (n x m x m) + (n x 1 x m) + (n x m x 1) -> (n x m x m)
		layer += lin_layer1 + lin_layer2
	# (n x m x m) + () -> (n x m x m)
	layer += biases
	return layer

#===============================================================
def bilinear_attention(layer1, layer2, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()+add_linear
	input2_size = layer2.get_shape().as_list()[-1]
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	original_layer1 = layer1
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
		noise_shape2 = tf.stack(layer_shape[:-2] + [1, input2_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
	if add_linear:
		ones = tf.ones(ones_shape)
		layer1 = tf.concat([layer1, ones], -1)
	
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])
	# (n x m x d) -> (n x m x d)
	layer2 = nn.reshape(layer2, [-1, bucket_size, input2_size])
	
	# (nm x d) * (d x d) -> (nm x d)
	attn = tf.matmul(layer1, weights)
	# (nm x d) -> (n x m x d)
	attn = nn.reshape(attn, [-1, bucket_size, input2_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	attn = tf.matmul(attn, layer2, transpose_b=True)
	# (n x m x m) -> (n x m x m)
	attn = nn.reshape(attn, layer_shape + [bucket_size])
	# (n x m x m) -> (n x m x m)
	soft_attn = tf.nn.softmax(attn)
	# (n x m x m) * (n x m x d) -> (n x m x d)
	weighted_layer1 = tf.matmul(soft_attn, original_layer1)
	
	return attn, weighted_layer1
	
#===============================================================
def diagonal_bilinear_attention(layer1, layer2, hidden_keep_prob=1., add_linear=True):
	""""""
	
	layer_shape = nn.get_sizes(layer1)
	bucket_size = layer_shape[-2]
	input1_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	ones_shape = tf.stack(layer_shape + [1])
	
	weights = tf.get_variable('Weights', shape=[input_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights2 = tf.get_variable('Weights2', shape=[input_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	original_layer1 = layer1
	if hidden_keep_prob < 1.:
		noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
	
	if add_linear:
		#(d) -> (d x 1)
		weights2 = tf.expand_dims(weights2, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_attn2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x 1) -> (nm x 1)
		lin_attn2 = tf.matmul(lin_attn2, weights2)
		# (nm x 1) -> (n x m)
		lin_attn2 = nn.reshape(lin_attn2, layer_shape)
		# (n x m) -> (n x 1 x m)
		lin_attn2 = tf.expand_dims(lin_attn2, axis=-2)
	
	# (n x m x d) -> (nm x d)
	layer1 = nn.reshape(layer1, [-1, input_size])
	# (n x m x d) -> (n x m x d)
	layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
	
	# (nm x d) * (d) -> (nm x d)
	attn = layer1 * weights
	# (nm x d) -> (n x m x d)
	attn = nn.reshape(attn, [-1, bucket_size, input_size])
	# (n x m x d) * (n x m x d) -> (n x m x m)
	attn = tf.matmul(attn, layer2, transpose_b=True)
	# (n x m x m) -> (n x m x m)
	attn = nn.reshape(attn, layer_shape + [bucket_size])
	if add_linear:
		# (n x m x m) + (n x 1 x m) -> (n x m x m)
		attn += lin_attn2
	# (n x m x m) -> (n x m x m)
	soft_attn = tf.nn.softmax(attn)
	# (n x m x m) * (n x m x d) -> (n x m x d)
	weighted_layer1 = tf.matmul(soft_attn, original_layer1)

	return attn, weighted_layer1

#===============================================================
def span_trilinear_discriminator(layer1, layer2, layer3, hidden_keep_prob=1., add_linear = True, tri_std=0.01, hidden_k=200):
	""""""
	# use outer tensor product for trilinear layer size of k x d x 1
	# pdb.set_trace()
	# print(tri_std)
	pred_shape = nn.get_sizes(layer1)
	layer_shape = nn.get_sizes(layer2)
	layer3_shape = nn.get_sizes(layer3)
	bucket_size = layer_shape[-2]
	input1_size = pred_shape.pop() + 1
	input2_size = layer_shape.pop() + 1
	# here add a third layer
	input3_size = layer3_shape.pop() + 1
	pred_ones_shape = tf.stack(pred_shape + [1])
	ones_shape = tf.stack(layer_shape + [1])
	ones3_shape = tf.stack(layer3_shape + [1])
	# (d x d x d) layer1=axis0, layer2=axis2, layer3=axis1

	weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, hidden_k])
                               # ,initializer=tf.random_normal_initializer(stddev=tri_std))
	weights2 = tf.get_variable('trilinear_Weights2', shape=[input2_size, hidden_k])
								 # ,initializer=tf.random_normal_initializer(stddev=tri_std))
	weights3 = tf.get_variable('trilinear_Weights3', shape=[input3_size, hidden_k])
								 # ,initializer=tf.random_normal_initializer(stddev=tri_std))

	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(pred_shape[:-1] + [1, input1_size - 1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size - 1])
		noise_shape3 = tf.stack(layer3_shape[:-1] + [1, input3_size - 1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape3)
	pred_ones = tf.ones(pred_ones_shape)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, pred_ones], -1)
	layer2 = tf.concat([layer2, ones], -1)
	ones3 = tf.ones(ones3_shape)
	layer3 = tf.concat([layer3, ones3], -1)
	# pdb.set_trace()
	# (n x p x d) * (d x k) -> (n x p x k)
	layer1_tmp = tf.tensordot(layer1, weights1, axes=[[-1], [0]])
	# (n x m x d) * (d x k) -> (n x m x k)
	layer2_tmp = tf.tensordot(layer2, weights2, axes=[[-1], [0]])
	# (n x m x d) * (d x k) -> (n x m x k)
	layer3_tmp = tf.tensordot(layer3, weights3, axes=[[-1], [0]])
	# (n x p x k) * (n x mb x k) -> (n x p x mb x k)
	layer12_tmp = tf.einsum('nak,nbk->nabk', layer1_tmp, layer2_tmp)
	# (n x p x mb x k) * (n x mc x k) -> (n x p x mb x mc)
	layer = tf.einsum('nabk,nck->nabc', layer12_tmp, layer3_tmp)
	# layer=tf.reduce_sum(layer123,1)
	return layer

# ===============================================================
def span_bilinear_discriminator(layer1, layer2, hidden_keep_prob=1., add_linear=True, target_model='CRF', tri_std=0.5):
	""""""
	# pdb.set_trace()
	pred_shape = nn.get_sizes(layer1)
	layer_shape = nn.get_sizes(layer2)

	pred_size = pred_shape[-2]
	input1_size = pred_shape.pop() + 1
	input2_size = layer_shape.pop() + 1
	pred_ones_shape = tf.stack(pred_shape + [1])
	ones_shape = tf.stack(layer_shape + [1])
	if target_model == 'CRF':
		weights = tf.get_variable('Weights', shape=[input1_size, input2_size],
								  initializer=tf.random_normal_initializer(stddev=tri_std))
	elif target_model == 'LBP':
		weights = tf.get_variable('Weights', shape=[input1_size, input2_size],
								  initializer=tf.random_normal_initializer(stddev=tri_std))
	else:
		weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.zeros_initializer)
	# weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.truncated_normal_initializer())
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))

	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(pred_shape[:-1] + [1, input1_size - 1])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size - 1])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
	pred_ones = tf.ones(pred_ones_shape)
	ones = tf.ones(ones_shape)
	layer1 = tf.concat([layer1, pred_ones], -1)
	layer2 = tf.concat([layer2, ones], -1)

	# (n x p x d) -> (np x d)
	layer1 = nn.reshape(layer1, [-1, input1_size])

	# (np x d) * (d x d) -> (nm x d)
	layer = tf.matmul(layer1, weights)  # here is the u matrix in the paper?
	# (np x d) -> (n x p x d)
	layer = nn.reshape(layer, [-1, pred_size, input2_size])
	if len(layer_shape) == 3:
		# (n x p x d) * (n x m x m x d) -> (n x p x m x m)
		layer = tf.einsum('npd,nabd->npab', layer, layer2)
	else:
		# (n x p x d) * (n x k x d) -> (n x p x k)
		layer = tf.einsum('npd,nkd->npk', layer, layer2)

	return layer

# ===============================================================
def span_diagonal_bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob=1., add_linear=True):
	""""""
	# here is token classifier
	pred_shape = nn.get_sizes(layer1)
	layer_shape = nn.get_sizes(layer2)
	pred_size = pred_shape[1]
	input1_size = pred_shape.pop()
	input2_size = layer_shape.pop()


	weights = tf.get_variable('Weights', shape=[input1_size, output_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input1_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input2_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape1 = tf.stack(pred_shape[:-1] + [1, input1_size])
		noise_shape2 = tf.stack(layer_shape[:-1] + [1, input2_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)

	if add_linear:  # why here do not use weights2?
		# (n x p x d) -> (np x d)
		lin_layer1 = nn.reshape(layer1, [-1, input1_size])
		# (np x d) * (d x o) -> (np x o)
		lin_layer1 = tf.matmul(lin_layer1, weights1)
		# (np x o) -> (n x p x o)
		lin_layer1 = nn.reshape(lin_layer1, pred_shape + [output_size])
		# (n x p x o) -> (n x p x 1 x 1 x o)
		lin_layer1 = tf.expand_dims(tf.expand_dims(lin_layer1, axis=-2), axis=-2)
		# (n x m x m x d) -> (nmm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input2_size])
		# (nmm x d) * (d x o) -> (nmm x o)
		lin_layer2 = tf.matmul(lin_layer2, weights1)
		# (nmm x o) -> (n x m x m x o)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape + [output_size])
		# (n x m x m x o) -> (n x 1 x m x m x o)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=1)

	# (n x p x d) -> (n x p x 1 x d)
	layer1 = nn.reshape(layer1, [-1, pred_size, 1, input1_size])

	# (d x o) -> (o x d)
	weights = tf.transpose(weights, [1, 0])

	# means every word in layer1 have m label?
	# (n x p x 1 x d) (*) (o x d) -> (n x p x o x d)
	layer = layer1 * weights

	# n x p x o x d * n x m x m x d -> n x p x m x m x o
	layer = tf.einsum('npod,nabd->npabo', layer, layer2)
	if add_linear:
		# (n x p x m x m x o) + (n x p x 1 x 1 x o) + (n x 1 x m x m x o) -> (n x p x m x m x o)
		layer += lin_layer1 + lin_layer2
	# (n x p x m x m x o) + (o) -> (n x p x m x m x o)
	layer += biases
	return layer

#===============================================================
def span_bilinear_classifier(layer1, layer2, role, hidden_keep_prob=1., add_linear=False, hidden_k=200, tri_std=0.01):
	# here is token classifier
	pred_shape = nn.get_sizes(layer1)
	layer_shape = nn.get_sizes(layer2)

	input1_size = pred_shape.pop()
	input2_size = layer_shape.pop()
	nr = role.get_shape().as_list()[0]
	role_size = role.get_shape().as_list()[-1]


	weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights2 = tf.get_variable('trilinear_Weights2', shape=[input2_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights4 = tf.get_variable('trilinear_Weights4', shape=[role_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))

	biases = tf.get_variable('Biases', shape=[nr], initializer=tf.zeros_initializer)


	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights4))

	if add_linear:
		weights5 = tf.get_variable('Weights5', shape=[input1_size, nr], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights6 = tf.get_variable('Weights6', shape=[input2_size, nr], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))

		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights5))
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights6))


	if hidden_keep_prob < 1.:
		noise_pred_shape = tf.stack(pred_shape[:-1] + [1, input1_size])
		noise_shape = tf.stack(layer_shape[:-1] + [1, input2_size])

		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_pred_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)


	if add_linear:  # why here do not use weights2?
		# (n x p x d) -> (np x d)
		lin_layer1 = nn.reshape(layer1, [-1, input1_size])
		# (np x d) * (d x o) -> (np x o)
		lin_layer1 = tf.matmul(lin_layer1, weights5)
		# (np x o) -> (n x p x o)
		lin_layer1 = nn.reshape(lin_layer1, pred_shape + [nr])
		# (n x p x o) -> (n x p x 1 x o )
		lin_layer1 = tf.expand_dims(lin_layer1, axis=2)

		# (n x m x d) -> (nm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input2_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer2 = tf.matmul(lin_layer2, weights6)
		# (nm x o) -> (n x m x o)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape + [nr])
		# (n x m x o) -> (n x 1 x m x o)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=1)


	# (n x p x d) * (d x k) -> (n x p x k)
	layer1 = tf.tensordot(layer1, weights1, axes=[[-1], [0]])
	# (n x m1 x d) * (d x k) -> (n x m1 x k)
	layer2 = tf.tensordot(layer2, weights2, axes=[[-1], [0]])
	# (nr x d) * (d x k) -> (nr x k)
	role = tf.matmul(role, weights4)


	# (n x m1 x k), (nr x k) -> (n x mc1 x nr x k)
	layer2_role = tf.einsum('nak,rk->nark', layer2, role)
	# (n x p x k), (n x m1 x nr x k) -> (n x p x m1 x nr)
	layer = tf.einsum('npk,nark->npar', layer1, layer2_role)


	if add_linear:
		# (n x p x k x o) + (n x p x 1 x o) + (n x 1 x m x o) -> (n x p x k x o)
		layer += lin_layer1 + lin_layer2

	# (n x p x k x nr) + (nr) -> (n x p x k x nr)
	layer += biases
	return layer

#===============================================================
def diagonal_span_bilinear_classifier(layer1, layer2, output_size, hidden_keep_prob=1., add_linear=False):
	# here is token classifier
	pred_shape = nn.get_sizes(layer1)
	layer_shape = nn.get_sizes(layer2)
	pred_size = pred_shape[-2]
	bucket_size = layer_shape[-2]

	input1_size = pred_shape.pop()
	input2_size = layer_shape.pop()

	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	# ones_shape = tf.stack(layer_shape + [1])

	weights = tf.get_variable('Weights', shape=[input_size, output_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape_pred = tf.stack(pred_shape[:-1] + [1, input_size])
		noise_shape_layer = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape_pred)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape_layer)

	if add_linear:  # why here do not use weights2?
		# (n x p x d) -> (np x d)
		lin_layer1 = nn.reshape(layer1, [-1, input_size])
		# (np x d) * (d x o) -> (np x o)
		lin_layer1 = tf.matmul(lin_layer1, weights1)
		# (np x o) -> (n x p x o)
		lin_layer1 = nn.reshape(lin_layer1, pred_shape + [output_size])
		# (n x p x o) -> (n x p x o x 1)
		lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
		# (n x k x d) -> (nk x d)
		lin_layer2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x o) -> (nk x o)
		lin_layer2 = tf.matmul(lin_layer2, weights1)
		# (nk x o) -> (n x k x o)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape + [output_size])
		# (n x k x o) -> (n x o x k)
		lin_layer2 = tf.transpose(lin_layer2, [0, 2, 1])
		# (n x o x k) -> (n x 1 x o x k)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=-3)

	# (n x p x d) -> (n x p x 1 x d)
	layer1 = nn.reshape(layer1, [-1, pred_size, 1, input_size])
	# (n x k x d) -> (n x k x d)
	layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
	# (d x o) -> (o x d)
	weights = tf.transpose(weights, [1, 0])
	# (o) -> (o x 1)
	biases = nn.reshape(biases, [output_size, 1])
	# (n x p x 1 x d) (*) (o x d) -> (n x p x o x d)
	layer = layer1 * weights
	# (n x p x o x d) -> (n x po x d)
	layer = nn.reshape(layer, [-1, pred_size * output_size, input_size])
	# (n x po x d) * (n x k x d) -> (n x po x k)
	layer = tf.matmul(layer, layer2, transpose_b=True)
	# (n x po x k) -> (n x p x o x k)
	layer = nn.reshape(layer, pred_shape + [output_size, bucket_size])
	if add_linear:
		# (n x p x o x k) + (n x p x o x 1) + (n x 1 x o x k) -> (n x p x o x k)
		layer += lin_layer1 + lin_layer2
	# (n x m x o x m) + (o x 1) -> (n x m x o x m)
	layer += biases
	# n x p x k x o
	layer = tf.transpose(layer, [0,1,3,2])
	return layer

#===============================================================
def span_diagonal_trilinear_classifier(layer1, layer2, layer3, output_size, hidden_keep_prob=1., add_linear=False):
	# here is token classifier
	pred_shape = nn.get_sizes(layer1)
	pred_size = pred_shape[-2]
	layer_shape = nn.get_sizes(layer2)
	bucket_size = layer_shape[-2]
	input1_size = pred_shape.pop()
	input_size = layer_shape.pop()
	input2_size = layer2.get_shape().as_list()[-1]
	input3_size = layer3.get_shape().as_list()[-1]
	assert input1_size == input2_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	assert input2_size == input3_size, "Inputs to diagonal_full_bilinear_classifier don't match"
	input_size = input1_size
	

	weights = tf.get_variable('Weights', shape=[output_size, input_size], initializer=tf.zeros_initializer)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
	if add_linear:
		weights1 = tf.get_variable('Weights1', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights2 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
		weights3 = tf.get_variable('Weights2', shape=[input_size, output_size], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	biases = tf.get_variable('Biases', shape=[output_size], initializer=tf.zeros_initializer)
	if hidden_keep_prob < 1.:
		noise_shape_pred = tf.stack(pred_shape[:-1] + [1, input_size])
		noise_shape = tf.stack(layer_shape[:-1] + [1, input_size])
		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_shape_pred)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape)

	if add_linear:  # why here do not use weights2?
		# (n x m x d) -> (nm x d)
		lin_layer1 = nn.reshape(layer1, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer1 = tf.matmul(lin_layer1, weights1)
		# (nm x o) -> (n x m x o)
		lin_layer1 = nn.reshape(lin_layer1, pred_shape + [output_size])
		# (n x m x o) -> (n x m x o x 1)
		lin_layer1 = tf.expand_dims(lin_layer1, axis=-1)
		# (n x m x d) -> (nm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer2 = tf.matmul(lin_layer2, weights2)
		# (nm x o) -> (n x m x o)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer2 = tf.transpose(lin_layer2, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer2 = tf.expand_dims(lin_layer2, axis=-3)

		# (n x m x d) -> (nm x d)
		lin_layer3 = nn.reshape(layer3, [-1, input_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer3 = tf.matmul(lin_layer3, weights3)
		# (nm x o) -> (n x m x o)
		lin_layer3 = nn.reshape(lin_layer3, layer_shape + [output_size])
		# (n x m x o) -> (n x o x m)
		lin_layer3 = tf.transpose(lin_layer3, [0, 2, 1])
		# (n x o x m) -> (n x 1 x o x m)
		lin_layer3 = tf.expand_dims(lin_layer3, axis=-3)

	# n x p x d, n x m x d -> n x p x m x d
	layer3 = tf.einsum('npk,nmk->npmk', layer1, layer3)
	layer3 = nn.reshape(layer3, [-1, pred_size * bucket_size, 1, input_size])
	# n x m x d
	layer2 = nn.reshape(layer2, [-1, bucket_size, input_size])
	# (n x p*m x 1 x d) (*) (o x d) -> (n x p*m x o x d)
	layer3 = layer3 * weights
	# (n x p*m x o x d) -> (n x pmo x d)
	layer = nn.reshape(layer3, [-1, pred_size * bucket_size * output_size, input_size])
	# (n x pmo x d) * (n x m x d) -> (n x pmo x m)
	layer = tf.matmul(layer, layer2, transpose_b=True)
	# (n x pmo x m) -> (n x p x m x o x m)
	layer = nn.reshape(layer, [-1, pred_size, bucket_size, output_size, bucket_size])  # n x p x c x o x c

	if add_linear:
		# (n x m x o x m) + (n x 1 x o x m) + (n x m x o x 1) -> (n x m x o x m)
		layer += lin_layer1 + lin_layer2

	# (n x p x c x o x c) -> (n x p x c x c x o)
	layer = tf.transpose(layer, perm=[0, 1, 2, 4, 3])  # n x p x c x c x o
	# (n x p x m x m x o) + (o) -> (n x p x m x m x o)
	layer += biases

	return layer

#===============================================================
def span_trilinear_classifier(layer1, layer2, layer3, role, hidden_keep_prob=1., add_linear=False, hidden_k=200, tri_std=0.01):
	# here is token classifier
	pred_shape = nn.get_sizes(layer1)
	layer_shape = nn.get_sizes(layer2)

	input1_size = pred_shape.pop()
	input2_size = layer_shape.pop()
	input3_size = layer3.get_shape().as_list()[-1]
	nr = role.get_shape().as_list()[0]
	role_size = role.get_shape().as_list()[-1]


	weights1 = tf.get_variable('trilinear_Weights1', shape=[input1_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights2 = tf.get_variable('trilinear_Weights2', shape=[input2_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights3 = tf.get_variable('trilinear_Weights3', shape=[input3_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))
	weights4 = tf.get_variable('trilinear_Weights4', shape=[role_size, hidden_k],
							   initializer=tf.random_normal_initializer(stddev=tri_std))

	biases = tf.get_variable('Biases', shape=[nr], initializer=tf.zeros_initializer)


	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights3))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights4))

	if add_linear:
		weights5 = tf.get_variable('Weights5', shape=[input1_size, nr], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights1))
		weights6 = tf.get_variable('Weights6', shape=[input2_size, nr], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights2))
		weights7 = tf.get_variable('Weights7', shape=[input3_size, nr], initializer=tf.zeros_initializer)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights5))
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights6))
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights7))

	if hidden_keep_prob < 1.:
		noise_pred_shape = tf.stack(pred_shape[:-1] + [1, input1_size])
		noise_shape = tf.stack(layer_shape[:-1] + [1, input2_size])

		layer1 = nn.dropout(layer1, hidden_keep_prob, noise_shape=noise_pred_shape)
		layer2 = nn.dropout(layer2, hidden_keep_prob, noise_shape=noise_shape)
		layer3 = nn.dropout(layer3, hidden_keep_prob, noise_shape=noise_shape)

	if add_linear:  # why here do not use weights2?
		# (n x p x d) -> (np x d)
		lin_layer1 = nn.reshape(layer1, [-1, input1_size])
		# (np x d) * (d x o) -> (np x o)
		lin_layer1 = tf.matmul(lin_layer1, weights5)
		# (np x o) -> (n x p x o)
		lin_layer1 = nn.reshape(lin_layer1, pred_shape + [nr])
		# (n x p x o) -> (n x p x 1 x 1 x o )
		lin_layer1 = tf.expand_dims(tf.expand_dims(lin_layer1, axis=2), axis=2)

		# (n x m x d) -> (nm x d)
		lin_layer2 = nn.reshape(layer2, [-1, input2_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer2 = tf.matmul(lin_layer2, weights6)
		# (nm x o) -> (n x m x o)
		lin_layer2 = nn.reshape(lin_layer2, layer_shape + [nr])
		# (n x m x o) -> (n x 1 x m x 1 x o)
		lin_layer2 = tf.expand_dims(tf.expand_dims(lin_layer2, axis=1),axis=-2)

		# (n x m x d) -> (nm x d)
		lin_layer3 = nn.reshape(layer3, [-1, input3_size])
		# (nm x d) * (d x o) -> (nm x o)
		lin_layer3 = tf.matmul(lin_layer3, weights7)
		# (nm x o) -> (n x m x o)
		lin_layer3 = nn.reshape(lin_layer3, layer_shape + [nr])
		# (n x m x o) -> (n x 1 x 1 x m x o)
		lin_layer3 = tf.expand_dims(tf.expand_dims(lin_layer3, axis=1),axis=1)

	# (n x p x d) * (d x k) -> (n x p x k)
	layer1 = tf.tensordot(layer1, weights1, axes=[[-1], [0]])
	# (n x m1 x d) * (d x k) -> (n x m1 x k)
	layer2 = tf.tensordot(layer2, weights2, axes=[[-1], [0]])
	# (n x m2 x d) * (d x k) -> (n x m2 x k)
	layer3 = tf.tensordot(layer3, weights3, axes=[[-1], [0]])
	# (nr x d) * (d x k) -> (nr x k)
	role = tf.matmul(role, weights4)


	# (n x m1 x k), (nr x k) -> (n x mc1 x nr x k)
	layer2_role = tf.einsum('nak,rk->nark', layer2, role)
	# (n x p x k), (n x m1 x nr x k) -> (n x p x m1 x nr x k)
	layer1_layer2 = tf.einsum('npk,nark->npark', layer1, layer2_role)
	# (n x p x m1 x nr x k), (n x m2 x k) -> (n x p x m1 x m2 x nr)
	layer = tf.einsum('npark,nbk->npabr', layer1_layer2, layer3)

	if add_linear:
		# (n x p x m x m x o) + (n x p x 1 x 1 x o) + (n x 1 x m x 1 x o) + (n x 1 x 1 x m x o) -> (n x p x m x m x o)
		layer += lin_layer1 + lin_layer2 + lin_layer3

	# (n x p x m x m x nr) + (nr) -> (n x p x m x m x nr)
	layer += biases
	return layer