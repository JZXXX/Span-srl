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

import re
import os
import pickle as pkl
import curses
import codecs
import time
import numpy as np
import tensorflow as tf
from parser.graph_outputs import GraphOutputs, TrainOutputs, DevOutputs
from parser.structs import conllu_dataset
from parser.base_network import BaseNetwork
from parser.neural import nn, nonlin, embeddings, recurrent, classifiers
from parser.structs.vocabs.pointer_generator import PointerGenerator
import pdb


# ***************************************************************
class ArgumentDetectionNetwork(BaseNetwork):
	""""""

	# =============================================================
	def build_graph(self, input_network_outputs={}, reuse=True, debug=False, nornn=False):
		""""""
		# pdb.set_trace()

		with tf.variable_scope('Embeddings'):

			input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs if not input_vocab.classname == 'PredIndexVocab']
			# pdb.set_trace()

			layer = tf.concat(input_tensors, 2)  # batch*sentence*feature? or batch* sentence^2*feature?
		# pdb.set_trace()

		n_nonzero = tf.to_float(tf.count_nonzero(layer, axis=-1, keepdims=True))
		batch_size, bucket_size, input_size = nn.get_sizes(layer)
		layer *= input_size / (n_nonzero + tf.constant(1e-12))

		token_weights = nn.greater(self.id_vocab.placeholder, 0)  # find sentence length
		tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
		seq_lengths = tokens_per_sequence + 1  # batch size list of sentence length

		n_tokens = tf.reduce_sum(tokens_per_sequence)
		n_sequences = tf.count_nonzero(tokens_per_sequence)
		output_fields = {vocab.field: vocab for vocab in self.output_vocabs}
		outputs = {}
		# pdb.set_trace()
		root_weights = token_weights + (1 - nn.greater(tf.range(bucket_size), 0))
		token_weights3D = tf.expand_dims(token_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
		token_weights3D = token_weights3D * tf.transpose(token_weights3D,[0,2,1])


		tokens = {'n_tokens': n_tokens,
				  'tokens_per_sequence': tokens_per_sequence,
				  'input_tensors': input_tensors,
				  'token_weights': token_weights,
				  'token_weights3D': token_weights3D,
				  'n_sequences': n_sequences}
		# ===========================================================================================


		conv_keep_prob = 1. if reuse else self.conv_keep_prob
		recur_keep_prob = 1. if reuse else self.recur_keep_prob
		recur_include_prob = 1. if reuse else self.recur_include_prob
		# R=BiLSTM(X)
		# pdb.set_trace()
		for i in six.moves.range(self.n_layers):
			conv_width = self.first_layer_conv_width if not i else self.conv_width
			# '''
			if not nornn and not self.nornn:
				with tf.variable_scope('RNN-{}'.format(i)):
					layer, sentence_feat = recurrent.directed_RNN(layer, self.recur_size, seq_lengths,
																  bidirectional=self.bidirectional,
																  recur_cell=self.recur_cell,
																  conv_width=conv_width,
																  recur_func=self.recur_func,
																  conv_keep_prob=conv_keep_prob,
																  recur_include_prob=recur_include_prob,
																  recur_keep_prob=recur_keep_prob,
																  cifg=self.cifg,
																  highway=self.highway,
																  highway_func=self.highway_func,
																  bilin=self.bilin)
		# '''
		# pdb.set_trace()

		# layers
		with tf.variable_scope('Classifiers'):

			if 'argument' in output_fields:
				vocab = output_fields['argument']


				unlabeled_outputs = vocab.get_bilinear_discriminator(layer,token_weights3D,reuse=reuse)
				outputs['argument'] = unlabeled_outputs
				self._evals.add('argument')

		return outputs, tokens


	# @staticmethod
	# #==============================================================
	# def get_top_args(self, outputs):
	# 	for output in outputs:
	# 		pass  # we just need to grab one
	# 	top_args_idx = tf.stop_gradient(output['top_args'])
	#
	# 	return  top_args_idx


	@property
	def syn_weight(self):
		try:
			return self._config.getfloat(self, 'syn_weight')
		except:
			return 1.

	@property
	def syntactic(self):
		try:
			return self._config.getboolean(self, 'syntactic')
		except:
			return False

	def layer_mask(self, vocab):
		try:
			return self._config.getboolean(vocab, 'layer_mask')
		except:
			return False

	@property
	def mask(self):
		try:
			return self._config.getboolean(self, 'mask')
		except:
			return False

	@property
	def mask_decoder(self):
		try:
			return self._config.getboolean(self, 'mask_decoder')
		except:
			return False

	@property
	def no_flag(self):
		try:
			return self._config.getboolean(self, 'no_flag')
		except:
			return False

	@property
	def sum_pos(self):
		return self._config.getboolean(self, 'sum_pos')

	@property
	def separate_prediction(self):
		try:
			return self._config.getboolean(self, 'separate_prediction')
		except:
			return False

	@property
	def two_gpu(self):
		try:
			return self._config.getboolean(self, 'two_gpu')
		except:
			return False

	@property
	def nornn(self):
		try:
			return self._config.getboolean(self, 'nornn')
		except:
			return False

	@property
	def predict_attribute(self):
		try:
			return self._config.getboolean(self, 'predict_attribute')
		except:
			return False
