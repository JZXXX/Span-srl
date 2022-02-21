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

import os
import codecs
from collections import Counter

import numpy as np
import tensorflow as tf

from parser.structs.vocabs.base_vocabs import CountVocab
from . import conllu_vocabs as cv

from parser.neural import nn, nonlin, embeddings, classifiers
import pdb
import copy
#***************************************************************
class TokenVocab(CountVocab):
	""""""

	_save_str = 'tokens'

	#=============================================================
	def __init__(self, *args, **kwargs):
		""""""
		self.BOS_STR = '<bos>'
		self.BOS_IDX = 999998
		self.EOS_STR = '<eos>'
		self.EOS_IDX = 999999
		super(TokenVocab, self).__init__(*args, **kwargs)
		return

	#=============================================================
	def get_input_tensor(self, inputs=None, embed_keep_prob=None, nonzero_init=True, variable_scope=None, reuse=True):
		""""""

		if inputs is None:
			inputs = self.placeholder
		embed_keep_prob = 1 if reuse else (embed_keep_prob or self.embed_keep_prob)

		with tf.variable_scope(variable_scope or self.classname):
			layer = embeddings.token_embedding_lookup(len(self), self.embed_size,
																				inputs,
																				nonzero_init=nonzero_init,
																				reuse=reuse)
			if embed_keep_prob < 1:
				layer = self.drop_func(layer, embed_keep_prob)
		return layer

	#=============================================================
	# TODO confusion matrix
	def get_output_tensor(self, predictions, reuse=True):
		""""""

		embed_keep_prob = 1 if reuse else self.embed_keep_prob

		with tf.variable_scope(self.classname):
			layer = embeddings.token_embedding_lookup(len(self), self.embed_size,
																				predictions,
																				reuse=reuse)
			if embed_keep_prob < 1:
				layer = self.drop_func(layer, embed_keep_prob)
		return layer

	#=============================================================
	def get_linear_classifier(self, layer, token_weights, last_output=None, variable_scope=None, reuse=False):
		""""""

		if last_output is not None:
			n_layers = 0
			layer = last_output['hidden_layer']
			recur_layer = last_output['recur_layer']
		else:
			n_layers = self.n_layers
			recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		with tf.variable_scope(variable_scope or self.classname):
			for i in six.moves.range(0, n_layers):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, self.hidden_size,
																		hidden_func=self.hidden_func,
																		hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('Classifier'):
				logits = classifiers.linear_classifier(layer, len(self), hidden_keep_prob=hidden_keep_prob)
		targets = self.placeholder

		#-----------------------------------------------------------
		# Compute probabilities/cross entropy
		# (n x m x c) -> (n x m x c)
		probabilities = tf.nn.softmax(logits)
		# (n x m), (n x m x c), (n x m) -> ()
		loss = tf.losses.sparse_softmax_cross_entropy(targets, logits, weights=token_weights)

		#-----------------------------------------------------------
		# Compute predictions/accuracy
		# (n x m x c) -> (n x m)
		predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
		# (n x m) (*) (n x m) -> (n x m)
		correct_tokens = nn.equal(targets, predictions) * token_weights
		# (n x m) -> (n)
		tokens_per_sequence = tf.reduce_sum(token_weights, axis=-1)
		# (n x m) -> (n)
		correct_tokens_per_sequence = tf.reduce_sum(correct_tokens, axis=-1)
		# (n), (n) -> (n)
		correct_sequences = nn.equal(tokens_per_sequence, correct_tokens_per_sequence)

		#-----------------------------------------------------------
		# Populate the output dictionary
		outputs = {}
		outputs['recur_layer'] = recur_layer
		outputs['hidden_layer'] = layer
		outputs['targets'] = targets
		outputs['probabilities'] = probabilities
		outputs['loss'] = loss

		outputs['predictions'] = predictions
		outputs['n_correct_tokens'] = tf.reduce_sum(correct_tokens)
		outputs['n_correct_sequences'] = tf.reduce_sum(correct_sequences)
		return outputs

	#=============================================================
	def get_sampled_linear_classifier(self, layer, n_samples, token_weights=None, variable_scope=None, reuse=False):
		""""""

		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		with tf.variable_scope(variable_scope or self.classname):
			for i in six.moves.range(0, self.n_layers):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, self.hidden_size,
																		hidden_func=self.hidden_func,
																		hidden_keep_prob=hidden_keep_prob)
			batch_size, bucket_size, input_size = nn.get_sizes(layer)
			layer = nn.dropout(layer, hidden_keep_prob, noise_shape=[batch_size, 1, input_size])
			layer = nn.reshape(layer, [-1, input_size])


			with tf.variable_scope('Classifier'):
				# (s)
				samples, _, _ = tf.nn.log_uniform_candidate_sampler(
					nn.zeros([bucket_size,1], dtype=tf.int64),
					1, n_samples, unique=True, range_max=len(self))
				with tf.device('/gpu:1'):
					weights = tf.get_variable('Weights', shape=[len(self), input_size], initializer=tf.zeros_initializer)
					biases = tf.get_variable('Biases', shape=len(self), initializer=tf.zeros_initializer)
					tf.add_to_collection('non_save_variables', weights)
					tf.add_to_collection('non_save_variables', biases)

					# (nm x 1)
					targets = nn.reshape(self.placeholder, [-1, 1])
					# (1 x s)
					samples = tf.expand_dims(samples, 0)
					# (nm x s)
					samples = tf.to_int32(nn.tile(samples, [batch_size*bucket_size, 1]))
					# (nm x s)
					sample_weights = tf.to_float(nn.not_equal(samples, targets))
					# (nm x 1+s)
					cands = tf.stop_gradient(tf.concat([targets, samples], axis=-1))
					# (nm x 1), (nm x s) -> (nm x 1+s)
					cand_weights = tf.stop_gradient(tf.concat([nn.ones([batch_size*bucket_size, 1]), sample_weights], axis=-1))
					# (c x d), (nm x 1+s) -> (nm x 1+s x d)
					weights = tf.nn.embedding_lookup(weights, cands)
					# (c), (nm x 1+s) -> (nm x 1+s)
					biases = tf.nn.embedding_lookup(biases, cands)
					# (n x m x d) -> (nm x d x 1)
					layer_reshaped = nn.reshape(layer, [-1, input_size, 1])
					# (nm x 1+s x d) * (nm x d x 1) -> (nm x 1+s x 1)
					logits = tf.matmul(weights, layer_reshaped)
					# (nm x 1+s x 1) -> (nm x 1+s)
					logits = tf.squeeze(logits, -1)

					#-----------------------------------------------------------
					# Compute probabilities/cross entropy
					# (nm x 1+s)
					logits = logits - tf.reduce_max(logits, axis=-1, keep_dims=True)
					# (nm x 1+s)
					exp_logits = tf.exp(logits) * cand_weights
					# (nm x 1)
					exp_logit_sum = tf.reduce_sum(exp_logits, axis=-1, keep_dims=True)
					# (nm x 1+s)
					probabilities = exp_logits / exp_logit_sum
					# (nm x 1+s) -> (n x m x 1+s)
					probabilities = nn.reshape(probabilities, [batch_size, bucket_size, 1+n_samples])
					# (nm x 1+s) -> (n x m x 1+s)
					samples = nn.reshape(samples, [batch_size, bucket_size, 1+n_samples])
					# (nm x 1+s) -> (nm x 1), (nm x s)
					target_logits, _ = tf.split(logits, [1, n_samples], axis=1)
					# (nm x 1) - (nm x 1) -> (nm x 1)
					loss = tf.log(exp_logit_sum) - target_logits
					# (n x m) -> (nm x 1)
					token_weights1D = tf.to_float(nn.reshape(token_weights, [-1,1]))
					# (nm x 1) -> ()
					loss = tf.reduce_sum(loss*token_weights1D) / tf.reduce_sum(token_weights1D)

					#-----------------------------------------------------------
					# Compute predictions/accuracy
					# (nm x 1+s) -> (n x m x 1+s)
					logits = nn.reshape(logits, [batch_size, bucket_size, -1])
					# (n x m x 1+s) -> (n x m)
					predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
					# (n x m) (*) (n x m) -> (n x m)
					correct_tokens = nn.equal(predictions, 0) * token_weights
					# (n x m) -> (n)
					tokens_per_sequence = tf.reduce_sum(token_weights, axis=-1)
					# (n x m) -> (n)
					correct_tokens_per_sequence = tf.reduce_sum(correct_tokens, axis=-1)
					# (n), (n) -> (n)
					correct_sequences = nn.equal(tokens_per_sequence, correct_tokens_per_sequence)

					#-----------------------------------------------------------
					# Populate the output dictionary
					outputs = {}
					outputs['recur_layer'] = recur_layer
					outputs['targets'] = targets
					outputs['probabilities'] = tf.tuple([samples, probabilities])
					outputs['loss'] = loss

					outputs['predictions'] = predictions
					outputs['n_correct_tokens'] = tf.reduce_sum(correct_tokens)
					outputs['n_correct_sequences'] = tf.reduce_sum(correct_sequences)
		return outputs

	#=============================================================
	def get_bilinear_classifier(self, layer, outputs, token_weights, variable_scope=None, reuse=False):
		""""""

		layer1 = layer2 = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		hidden_func = self.hidden_func
		hidden_size = self.hidden_size
		add_linear = self.add_linear
		with tf.variable_scope(variable_scope or self.classname):
			for i in six.moves.range(0, self.n_layers-1):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, 2*hidden_size,
																		hidden_func=hidden_func,
																		hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top'):
				layers = classifiers.hiddens(layer, 2*[hidden_size],
																		hidden_func=hidden_func,
																		hidden_keep_prob=hidden_keep_prob)
			layer1, layer2 = layers.pop(0), layers.pop(0)

			with tf.variable_scope('Classifier'):
				if self.diagonal:
					logits = classifiers.diagonal_bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					logits = classifiers.bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				bucket_size = tf.shape(layer)[-2]

				#-------------------------------------------------------
				# Process the targets
				# (n x m)
				# pdb.set_trace()
				label_targets = self.placeholder
				unlabeled_predictions = outputs['unlabeled_predictions']
				unlabeled_targets = outputs['unlabeled_targets']
				# (n x m) -> (n x m x m)
				unlabeled_predictions = tf.one_hot(unlabeled_predictions, bucket_size)
				unlabeled_targets = tf.one_hot(unlabeled_targets, bucket_size)
				# (n x m x m) -> (n x m x m x 1)
				unlabeled_predictions = tf.expand_dims(unlabeled_predictions, axis=-1)
				unlabeled_targets = tf.expand_dims(unlabeled_targets, axis=-1)

				#-------------------------------------------------------
				# Process the logits
				# We use the gold heads for computing the label score and the predicted
				# heads for computing the unlabeled attachment score
				# (n x m x c x m) -> (n x m x m x c)
				transposed_logits = tf.transpose(logits, [0,1,3,2])
				# (n x m x c x m) * (n x m x m x 1) -> (n x m x c x 1)
				predicted_logits = tf.matmul(logits, unlabeled_predictions)
				oracle_logits = tf.matmul(logits, unlabeled_targets)
				# (n x m x c x 1) -> (n x m x c)
				predicted_logits = tf.squeeze(predicted_logits, axis=-1)
				oracle_logits = tf.squeeze(oracle_logits, axis=-1)

				#-------------------------------------------------------
				# Compute probabilities/cross entropy
				# (n x m x m) -> (n x m x m x 1)
				head_probabilities = tf.expand_dims(tf.stop_gradient(outputs['probabilities']), axis=-1)
				# (n x m x m x c) -> (n x m x m x c)
				label_probabilities = tf.nn.softmax(transposed_logits)
				# (n x m), (n x m x c), (n x m) -> ()
				label_loss = tf.losses.sparse_softmax_cross_entropy(label_targets, oracle_logits, weights=token_weights)

				#-------------------------------------------------------
				# Compute predictions/accuracy
				# (n x m x c) -> (n x m)
				label_predictions = tf.argmax(predicted_logits, axis=-1, output_type=tf.int32)
				label_oracle_predictions = tf.argmax(oracle_logits, axis=-1, output_type=tf.int32)
				# (n x m) (*) (n x m) -> (n x m)
				correct_label_tokens = nn.equal(label_targets, label_oracle_predictions) * token_weights
				correct_tokens = nn.equal(label_targets, label_predictions) * outputs['correct_unlabeled_tokens']

				# (n x m) -> (n)
				tokens_per_sequence = tf.reduce_sum(token_weights, axis=-1)
				# (n x m) -> (n)
				correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=-1)
				correct_tokens_per_sequence = tf.reduce_sum(correct_tokens, axis=-1)
				# (n), (n) -> (n)
				correct_label_sequences = nn.equal(tokens_per_sequence, correct_label_tokens_per_sequence)
				correct_sequences = nn.equal(tokens_per_sequence, correct_tokens_per_sequence)

		#-----------------------------------------------------------
		# Populate the output dictionary
		rho = self.loss_interpolation
		outputs['label_targets'] = label_targets
		# This way we can reconstruct the head_probabilities by exponentiating and summing along the last axis
		outputs['probabilities'] = label_probabilities * head_probabilities
		outputs['label_loss'] = label_loss
		outputs['loss'] = 2*((1-rho) * outputs['loss'] + rho * label_loss)

		outputs['label_predictions'] = label_predictions
		outputs['n_correct_label_tokens'] = tf.reduce_sum(correct_label_tokens)
		outputs['n_correct_label_sequences'] = tf.reduce_sum(correct_label_sequences)
		outputs['n_correct_tokens'] = tf.reduce_sum(correct_tokens)
		outputs['n_correct_sequences'] = tf.reduce_sum(correct_sequences)

		return outputs

	#=============================================================
	def get_bilinear_classifier_with_embeddings(self, layer, embeddings, token_weights, variable_scope=None, reuse=False):
		""""""

		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		with tf.variable_scope(variable_scope or self.classname):
			for i in six.moves.range(0, self.n_layers):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, self.hidden_size,
																		 hidden_func=self.hidden_func,
																		 hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('Classifier'):
				logits = classifiers.batch_bilinear_classifier(
					layer, embeddings, len(self),
					hidden_keep_prob=hidden_keep_prob,
					add_linear=self.add_linear)
				bucket_size = tf.shape(layer)[-2]
		targets = self.placeholder

		#-----------------------------------------------------------
		# Compute probabilities/cross entropy
		# (n x m x c) -> (n x m x c)
		probabilities = tf.nn.softmax(logits)
		# (n x m), (n x m x c), (n x m) -> ()
		loss = tf.losses.sparse_softmax_cross_entropy(targets, logits, weights=token_weights)

		#-----------------------------------------------------------
		# Compute predictions/accuracy
		# (n x m x c) -> (n x m)
		predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
		# (n x m) (*) (n x m) -> (n x m)
		correct_tokens = nn.equal(targets, predictions) * token_weights
		# (n x m) -> (n)
		tokens_per_sequence = tf.reduce_sum(token_weights, axis=-1)
		# (n x m) -> (n)
		correct_tokens_per_sequence = tf.reduce_sum(correct_tokens, axis=-1)
		# (n), (n) -> (n)
		correct_sequences = nn.equal(tokens_per_sequence, correct_tokens_per_sequence)

		#-----------------------------------------------------------
		# Populate the output dictionary
		outputs = {}
		outputs['recur_layer'] = recur_layer
		outputs['hidden_layer'] = layer
		outputs['targets'] = targets
		outputs['probabilities'] = probabilities
		outputs['loss'] = loss

		outputs['predictions'] = predictions
		outputs['n_correct_tokens'] = tf.reduce_sum(correct_tokens)
		outputs['n_correct_sequences'] = tf.reduce_sum(correct_sequences)
		return outputs

	#=============================================================
	def get_unfactored_bilinear_classifier(self, layer, unlabeled_targets, token_weights, variable_scope=None, reuse=False):
		""""""

		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		hidden_func = self.hidden_func
		hidden_size = self.hidden_size
		add_linear = self.add_linear
		with tf.variable_scope(variable_scope or self.classname):
			for i in six.moves.range(0, self.n_layers-1):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, 2*hidden_size,
																		hidden_func=hidden_func,
																		hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top'):
				layers = classifiers.hiddens(layer, 2*[hidden_size],
																		hidden_func=hidden_func,
																		hidden_keep_prob=hidden_keep_prob)
			layer1, layer2 = layers.pop(0), layers.pop(0)

			with tf.variable_scope('Classifier'):
				if self.diagonal:
					logits = classifiers.diagonal_bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					logits = classifiers.bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				bucket_size = tf.shape(layer)[-2]

				#-------------------------------------------------------
				# Process the targets
				# c (*) (n x m) + (n x m)
				#targets = len(self) * unlabeled_targets + self.placeholder
				targets = bucket_size * self.placeholder + unlabeled_targets

				#-------------------------------------------------------
				# Process the logits
				# (n x m x c x m) -> (n x m x cm)
				reshaped_logits = tf.reshape(logits, tf.stack([-1, bucket_size, bucket_size * len(self)]))

				#-------------------------------------------------------
				# Compute probabilities/cross entropy
				# (n x m x cm) -> (n x m x cm)
				probabilities = tf.nn.softmax(reshaped_logits)
				# (n x m x cm) -> (n x m x c x m)
				probabilities = tf.reshape(probabilities, tf.stack([-1, bucket_size, len(self), bucket_size]))
				# (n x m x c x m) -> (n x m x m x c)
				probabilities = tf.transpose(probabilities, [0,1,3,2])
				# (n x m), (n x m x cm), (n x m) -> ()
				loss = tf.losses.sparse_softmax_cross_entropy(targets, reshaped_logits, weights=token_weights)

				#-------------------------------------------------------
				# Compute predictions/accuracy
				# (n x m x cm) -> (n x m)
				predictions = tf.argmax(reshaped_logits, axis=-1, output_type=tf.int32)
				# (n x m), () -> (n x m)
				unlabeled_predictions = tf.mod(predictions, bucket_size)
				# (n x m) (*) (n x m) -> (n x m)
				correct_tokens = nn.equal(predictions, targets) * token_weights
				correct_unlabeled_tokens = nn.equal(unlabeled_predictions, unlabeled_targets) * token_weights

				# (n x m) -> (n)
				tokens_per_sequence = tf.reduce_sum(token_weights, axis=-1)
				# (n x m) -> (n)
				correct_tokens_per_sequence = tf.reduce_sum(correct_tokens, axis=-1)
				correct_unlabeled_tokens_per_sequence = tf.reduce_sum(correct_unlabeled_tokens, axis=-1)
				# (n), (n) -> (n)
				correct_sequences = nn.equal(tokens_per_sequence, correct_tokens_per_sequence)
				correct_unlabeled_sequences = nn.equal(tokens_per_sequence, correct_unlabeled_tokens_per_sequence)

		#-----------------------------------------------------------
		# Populate the output dictionary
		outputs = {}
		outputs['recur_layer'] = recur_layer
		outputs['unlabeled_targets'] = unlabeled_targets
		outputs['probabilities'] = probabilities
		outputs['unlabeled_loss'] = tf.constant(0.)
		outputs['loss'] = loss

		outputs['unlabeled_predictions'] = unlabeled_predictions
		outputs['label_predictions'] = predictions
		outputs['n_correct_unlabeled_tokens'] = tf.reduce_sum(correct_unlabeled_tokens)
		outputs['n_correct_unlabeled_sequences'] = tf.reduce_sum(correct_unlabeled_sequences)
		outputs['n_correct_tokens'] = tf.reduce_sum(correct_tokens)
		outputs['n_correct_sequences'] = tf.reduce_sum(correct_sequences)

		return outputs

	#=============================================================
	# TODO make this compatible with zipped files
	def count(self, train_conllus):
		""""""
		for train_conllu in train_conllus:
			# with codecs.open(train_conllu, encoding='utf-8', errors='ignore') as f:
			with open(train_conllu,encoding='utf8') as f:
				reader=f.readlines()
				for line in reader:
					line = line.strip()
					if line and not line.startswith('#'):
						line = line.split('\t')

						token = line[self.conllu_idx] # conllu_idx is provided by the CoNLLUVocab

						self._count(token)
		self.index_by_counts()
		return True

	def _count(self, token):
		if not self.cased:
			token = token.lower()
		self.counts[token] += 1
		return
	#=============================================================
	def get_bos(self):
		""""""

		return self.BOS_STR

	#=============================================================
	def get_eos(self):
		""""""

		return self.EOS_STR

	#=============================================================
	@property
	def diagonal(self):
		return self._config.getboolean(self, 'diagonal')
	@property
	def add_linear(self):
		return self._config.getboolean(self, 'add_linear')
	@property
	def loss_interpolation(self):
		return self._config.getfloat(self, 'loss_interpolation')
	@property
	def drop_func(self):
		drop_func = self._config.getstr(self, 'drop_func')
		if hasattr(embeddings, drop_func):
			return getattr(embeddings, drop_func)
		else:
			raise AttributeError("module '{}' has no attribute '{}'".format(embeddings.__name__, drop_func))
	@property
	def decomposition_level(self):
		return self._config.getint(self, 'decomposition_level')
	@property
	def n_layers(self):
		return self._config.getint(self, 'n_layers')
	@property
	def factorized(self):
		return self._config.getboolean(self, 'factorized')
	@property
	def hidden_size(self):
		return self._config.getint(self, 'hidden_size')
	@property
	def embed_size(self):
		return self._config.getint(self, 'embed_size')
	@property
	def embed_keep_prob(self):
		return self._config.getfloat(self, 'embed_keep_prob')
	@property
	def hidden_keep_prob(self):
		return self._config.getfloat(self, 'hidden_keep_prob')
	@property
	def hidden_func(self):
		hidden_func = self._config.getstr(self, 'hidden_func')
		if hasattr(nonlin, hidden_func):
			return getattr(nonlin, hidden_func)
		else:
			raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, hidden_func))
	@property
	def compare_precision(self):
		try:
			if self._config.get('DEFAULT', 'tb')=='ptb' or self._config.get('DEFAULT', 'tb')=='ctb':
				return True
			else:
				return False
		except:
			return False


# ***************************************************************
class SpanTokenVocab(TokenVocab):
	""""""

	_depth = -4

	# =============================================================
	def __init__(self, *args, **kwargs):
		""""""
		kwargs['placeholder_shape'] = [None, None, None, None]
		super(SpanTokenVocab, self).__init__(*args, **kwargs)
		return

	# =============================================================
	def _count(self, node):
		if node not in ('_', ''):
			node = node.split('|')
			for edge in node:
				edge = edge.split(':', 1)
				head, rel = edge[0], edge[1].split('-',1)[1]
				self.counts[rel] += 1
		return

	# =============================================================
	def add(self, token):
		""""""

		return self.index(token)

	# =============================================================
	# token should be: 1:(1,2)-rel|2:(3,4)-acl|5:(5,6)-dep
	def index(self, token):
		""""""

		nodes = []
		if token != '_':
			token = token.split('|')
			for edge in token:
				head, span_end, semrel = edge.split(':', 1)[0], edge.split(':',1)[1].split('-',1)[0].split(',')[1][:-1], edge.split(':',1)[1].split('-',1)[1]
				nodes.append((int(head), int(span_end), super(SpanTokenVocab, self).__getitem__(semrel)))
		return nodes

	# =============================================================
	# index should be [(1, 12, 23), (2, 4,12), (5, 2, 9)]
	def token(self, index):
		""""""

		nodes = []
		for (head, span_end, semrel) in index:
			nodes.append('{}:({})-{}'.format(head, span_end, super(SpanTokenVocab, self).__getitem__(semrel)))
		return '|'.join(nodes)

	# =============================================================
	def get_root(self):
		""""""

		return '_'

	# =============================================================
	def __getitem__(self, key):
		if isinstance(key, six.string_types):
			nodes = []
			if key != '_':
				token = key.split('|')
				for edge in token:
					head, span_end, semrel = edge.split(':', 1)[0], edge.split(':', 1)[1].split('-', 1)[0].split(',')[1][:-1], \
											 edge.split(':', 1)[1].split('-', 1)[1]
					nodes.append((int(head), int(span_end), super(SpanTokenVocab, self).__getitem__(semrel)))
			return nodes
		elif hasattr(key, '__iter__'):
			if len(key) > 0 and hasattr(key[0], '__iter__'):
				if len(key[0]) > 0 and isinstance(key[0][0], six.integer_types + (np.int32, np.int64)):
					nodes = []
					for (head, span_end, rel) in key:
						nodes.append(
							'{}:({})-{}'.format(head, span_end, super(SpanTokenVocab, self).__getitem__(rel)))
					return '|'.join(nodes)
				else:
					return [self[k] for k in key]
			else:
				return '_'
		else:
			raise ValueError(
				'key to SpanTokenVocab.__getitem__ must be (iterable of) strings or iterable of integers')


	# =============================================================
	def get_role_tensor(self, reuse=True):

		embed_keep_prob = 1 if reuse else self.embed_keep_prob

		with tf.variable_scope(self.classname):
			layer = embeddings.role_embedding_lookup(len(self), self.embed_size, reuse=reuse)
			# if embed_keep_prob < 1:
			# 	layer = embeddings.role_unkout(layer, embed_keep_prob)
		return layer


	# =============================================================
	def get_trilinear_classifier(self, layer, preds, outputs, token_weights, variable_scope=None, reuse=False, debug=False):
		""""""

		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear

		batch_size, bucket_size, input_size = nn.get_sizes(layer)
		one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
		layer_preds = tf.matmul(one_hot_pred, layer)
		pred_size = nn.get_sizes(layer_preds)[1]

		layer_args = layer
		layer_arge = layer


		with tf.variable_scope(variable_scope or self.field):

			with tf.variable_scope('Role'):
				layer_role = self.get_role_tensor(reuse=reuse)

			for i in six.moves.range(0, self.n_layers):
				with tf.variable_scope('ARGS-FC-%d' % i):
					layer_args = classifiers.hidden(layer_args, self.hidden_size,
													hidden_func=self.hidden_func,
													hidden_keep_prob=hidden_keep_prob)
				with tf.variable_scope('ARGE-FC-%d' % i):  # here is FNN? did not run
					layer_arge = classifiers.hidden(layer_arge, self.hidden_size,
													hidden_func=self.hidden_func,
													hidden_keep_prob=hidden_keep_prob)
				with tf.variable_scope('PRED-FC-%d' % i):
					layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
													 hidden_func=self.hidden_func,
													 hidden_keep_prob=hidden_keep_prob)
				if self.role_hidden:
					with tf.variable_scope('ROLE-FC-%d' % i):
						layer_role = classifiers.hidden(layer_role, self.hidden_size,
														 hidden_func=self.hidden_func,
														 hidden_keep_prob=hidden_keep_prob)


			with tf.variable_scope('Classifier'):
				if self.diagonal:
					logits = classifiers.span_diagonal_trilinear_classifier(
						layer_preds, layer_args, layer_arge, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					logits = classifiers.span_trilinear_classifier(
						layer_preds, layer_args, layer_arge, layer_role,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

		# -----------------------------------------------------------
		# Process the targets
		# (n x p x m x m)
		label_targets = self.placeholder
		unlabeled_predictions = outputs['unlabeled_predictions']
		unlabeled_targets = outputs['unlabeled_targets']
		top_args = False
		try:
			top_args_idx = outputs['top_args_idx']
			top_args = True
		except:
			pass


		# -----------------------------------------------------------
		# Compute the probabilities/cross entropy

		# (n x p x m x m x c) -> (n x p x m x m x c)
		label_probabilities = tf.nn.softmax(logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
		# (n x m x m), (n x m x m x c), (n x m x m) -> ()
		label_loss = tf.losses.sparse_softmax_cross_entropy(label_targets, logits,
															weights=token_weights * unlabeled_targets)

		# -----------------------------------------------------------
		# Compute the predictions/accuracy
		# (n x p x m x m x c) -> (n x p x m x m)
		# print('23333')
		predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
		# if top_k:
			# n x p x m x m
			# true_positives = nn.equal(label_targets, predictions)
			# true_positives = tf.reshape(tf.gather_nd(true_positives, top_args_idx),[batch_size,pred_size,-1])
			# true_positives = true_positives * unlabeled_predictions
		if top_args:
			# n x p x m x m
			true_positives = nn.equal(label_targets, predictions)
			# n x m x m x p
			true_positives = tf.transpose(true_positives, [0, 2, 3, 1])
			# n x k*p
			true_positives = tf.gather_nd(true_positives, top_args_idx)
			# n x p x k
			true_positives = tf.transpose(
				tf.reshape(true_positives, [batch_size, -1, pred_size]), [0, 2, 1])
			true_positives = true_positives * unlabeled_predictions
		else:
		# (n x p x m x m) (*) (n x p x m x m) -> (n x p x m x m)
			true_positives = nn.equal(label_targets, predictions) * unlabeled_predictions
		correct_label_tokens = nn.equal(label_targets, predictions) * unlabeled_targets
		# (n x m x m) -> ()
		n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
		n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
		n_true_positives = tf.reduce_sum(true_positives)
		n_correct_label_tokens = tf.reduce_sum(correct_label_tokens)
		# () - () -> ()
		n_false_positives = n_unlabeled_predictions - n_true_positives
		n_false_negatives = n_unlabeled_targets - n_true_positives
		# (n x m x m) -> (n)
		n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])
		# if top_k:
		# 	n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
		# else:
		if top_args:
			n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
		else:
			n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2, 3])
		n_correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=[1, 2, 3])
		# (n) x 2 -> ()
		n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
		n_correct_label_sequences = tf.reduce_sum(nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))

		# -----------------------------------------------------------
		# Populate the output dictionary
		rho = self.loss_interpolation
		outputs['label_predictions'] = predictions
		outputs['label_probabilities'] = label_probabilities
		outputs['label_targets'] = label_targets
		outputs['probabilities'] = label_probabilities
		outputs['label_loss'] = label_loss
		# outputs['label_logits'] = transposed_logits*tf.to_float(token_weights)
		# Combination of labeled loss and unlabeled loss
		outputs['loss'] = 2 * ((1 - rho) * outputs['unlabeled_loss'] + rho * label_loss)
		# outputs['loss'] = label_loss * self.loss_rel_interpolation + outputs['loss'] * self.loss_edge_interpolation
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences
		outputs['n_correct_label_tokens'] = n_correct_label_tokens
		outputs['n_correct_label_sequences'] = n_correct_label_sequences

		return outputs

	# ==============================================================
	def get_bilinear_classifier(self, layer, preds, outputs, token_weights, variable_scope=None, reuse=False):
		""""""
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear
		top_args_idx = outputs['top_args_idx']

		batch_size, bucket_size, input_size = nn.get_sizes(layer)
		# pdb.set_trace()
		one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
		layer_preds = tf.matmul(one_hot_pred, layer)
		pred_nums = nn.get_sizes(layer_preds)[1]

		# choose pre-difined args according to top_args_idx
		top_args_nums = nn.get_sizes(top_args_idx)[1]

		# layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
		#                         tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)
		#
		# # n x m x m x d -> n x k*d
		# top_layer_args = tf.gather_nd(layer_args, top_args_idx)
		# # n x k x d
		# top_layer_args = tf.reshape(top_layer_args, [batch_size, top_args_nums, input_size * 2])

		if self.span_diff:
			with tf.variable_scope(variable_scope or self.field):
				for i in six.moves.range(0, self.n_layers):
					with tf.variable_scope('LAYERS-FC-%d' % i):
						# n x m x d
						layer = classifiers.hidden(layer, self.hidden_size,
												   hidden_func=self.hidden_func,
												   hidden_keep_prob=hidden_keep_prob)
		batch_size, bucket_size, input_size = nn.get_sizes(layer)

		args_embed_type = self.args_embed_type

		# n x k x 3 -> n x k x 1
		top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
		top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
		top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])

		args_start_idx = tf.concat([top_args_idx_x, top_args_idx_y], -1)
		args_end_idx = tf.concat([top_args_idx_x, top_args_idx_z], -1)
		# n x k x d
		top_args_start = tf.reshape(tf.gather_nd(layer, args_start_idx), [batch_size, top_args_nums, input_size])
		top_args_end = tf.reshape(tf.gather_nd(layer, args_end_idx), [batch_size, top_args_nums, input_size])
		if args_embed_type == 'diff-sum':
			# n x k x 2d
			top_layer_args = tf.concat([top_args_start + top_args_end, top_args_end - top_args_start], axis=-1)
		elif args_embed_type == 'attention':
			# n x k x 3 -> n x k x 1
			# top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
			top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
			top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])
			attention_score = layer
			with tf.variable_scope(variable_scope or self.field):
				for i in six.moves.range(0, self.n_layers):
					with tf.variable_scope('ATTENTION-FC-%d' % i):
						# n x m x d
						attention_score = classifiers.hidden(attention_score, self.hidden_size,
															 hidden_func=self.hidden_func,
															 hidden_keep_prob=hidden_keep_prob)
				with tf.variable_scope('ATTENTION-TOP'):
					# n x m x 1
					attention_score = classifiers.hidden(attention_score, 1,
														 hidden_func=self.hidden_func,
														 hidden_keep_prob=hidden_keep_prob)

				# n x k x m
				attention_score = tf.transpose(tf.tile(attention_score, [1, 1, top_args_nums]), [0, 2, 1])
				mask = tf.expand_dims(tf.expand_dims(tf.range(bucket_size), 0), 0)
				mask = tf.cast(tf.tile(mask, [batch_size, top_args_nums, 1]), tf.int64)
				mask_tmp1 = tf.where(tf.greater_equal(mask, top_args_idx_y), tf.ones_like(mask), tf.zeros_like(mask))
				mask_tmp2 = tf.where(tf.greater_equal(top_args_idx_z, mask), tf.ones_like(mask), tf.zeros_like(mask))
				mask = tf.where(tf.greater(mask_tmp1 + mask_tmp2, 1), tf.zeros_like(mask), tf.ones_like(mask))
				# n x k x m  [1,3]->[-inf,0,0,0,-inf,-inf...]
				mask = tf.to_float(mask) * (-1e13)
				# n x k x m [-inf,score,score,score,-inf,-inf...] for every span
				attention_score = mask + attention_score
				attention_weight = tf.math.softmax(attention_score, -1)
				# n x k x m and n x m x d -> n x k x d
				top_layer_args = tf.einsum('nkm,nmd->nkd', attention_weight, layer)
				# concat start and end token embedding
				top_layer_args = tf.concat([top_args_start, top_args_end, top_layer_args], axis=-1)
		elif args_embed_type == 'endpoint':
			# n x k x 2d
			top_layer_args = tf.concat([top_args_start, top_args_end], axis=-1)
		elif args_embed_type == 'max-pooling':
			mask = tf.expand_dims(tf.expand_dims(tf.range(bucket_size), 0), 0)
			mask = tf.cast(tf.tile(mask, [batch_size, top_args_nums, 1]), tf.int64)
			mask_tmp1 = tf.where(tf.greater_equal(mask, top_args_idx_y), tf.ones_like(mask), tf.zeros_like(mask))
			mask_tmp2 = tf.where(tf.greater_equal(top_args_idx_z, mask), tf.ones_like(mask), tf.zeros_like(mask))
			mask = tf.where(tf.greater(mask_tmp1 + mask_tmp2, 1), tf.zeros_like(mask), tf.ones_like(mask))
			# n x k x m  [1,3]->[-inf,0,0,0,-inf,-inf...]
			mask = tf.to_float(mask) * (-1e13)
			# n x k x m [-inf,1,1,1,-inf,-inf...] for every span
			mask += tf.ones_like(mask)
			# n x k x m, n x m x d -> n x k x m x d
			top_layer_args = tf.einsum('nkm,nmd->nkmd', mask, layer)
			# n x k x m x d -> n x k x d
			top_layer_args = tf.reduce_max(top_layer_args, reduction_indices=-2, keep_dims=False)
		elif args_embed_type == 'coherent':
			top_args_start1, top_args_start2, top_args_start3, top_args_start4 = tf.split(top_args_start, tf.constant(
				[560, 560, 40, 40]), -1)
			top_args_end1, top_args_end2, top_args_end3, top_args_end4 = tf.split(top_args_end,
																				  tf.constant([560, 560, 40, 40]), -1)
			top_layer_args = tf.concat([top_args_start1, top_args_end2,
										tf.expand_dims(tf.einsum('nkd,nkd->nk', top_args_start3, top_args_end4), -1)],
									   -1)

		# n x p x m x m
		label_targets = self.placeholder
		# n x m x m x p
		label_targets_trans = tf.transpose(label_targets, [0, 2, 3, 1])
		# n x k*p
		top_label_targets = tf.gather_nd(label_targets_trans, top_args_idx)
		# n x p x k
		top_label_targets = tf.transpose(tf.reshape(top_label_targets, [batch_size, top_args_nums, pred_nums]),
										 [0, 2, 1])

		# n x m x m x p
		token_weights_trans = tf.transpose(token_weights, [0, 2, 3, 1])
		# n x k*p
		top_token_weights = tf.gather_nd(token_weights_trans, top_args_idx)
		# n x p x k
		top_token_weights = tf.transpose(tf.reshape(top_token_weights, [batch_size, top_args_nums, pred_nums]),
										 [0, 2, 1])

		with tf.variable_scope(variable_scope or self.field):

			with tf.variable_scope('Role'):
				layer_role = self.get_role_tensor(reuse=reuse)

			for i in six.moves.range(0, self.n_layers):
				with tf.variable_scope('ARGS-FC-%d' % i):
					top_layer_args = classifiers.hidden(top_layer_args, self.hidden_size,
														hidden_func=self.hidden_func,
														hidden_keep_prob=hidden_keep_prob)

				with tf.variable_scope('PRED-FC-%d' % i):
					layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
													 hidden_func=self.hidden_func,
													 hidden_keep_prob=hidden_keep_prob)
				if self.role_hidden:
					with tf.variable_scope('ROLE-FC-%d' % i):
						layer_role = classifiers.hidden(layer_role, self.hidden_size,
														hidden_func=self.hidden_func,
														hidden_keep_prob=hidden_keep_prob)

			with tf.variable_scope('Classifier'):

				if self.role_tensor:
					# n x p x k x role
					logits = classifiers.span_bilinear_classifier(
						layer_preds, top_layer_args, layer_role,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					# n x p x k x role
					logits = classifiers.diagonal_span_bilinear_classifier(
						layer_preds, top_layer_args, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

		# -----------------------------------------------------------
		# Process the targets

		unlabeled_predictions = outputs['unlabeled_predictions']
		top_unlabeled_targets = outputs['top_unlabeled_targets']
		unlabeled_targets = outputs['unlabeled_targets']

		# -----------------------------------------------------------
		# Compute the probabilities/cross entropy

		# (n x p x k x c) -> (n x p x k x c)
		label_probabilities = tf.nn.softmax(logits) * tf.to_float(tf.expand_dims(top_token_weights, axis=-1))
		# (n x k), (n x k x c), (n x k) -> ()
		label_loss = tf.losses.sparse_softmax_cross_entropy(top_label_targets, logits,
															weights=top_token_weights * top_unlabeled_targets)
		# pdb.set_trace()
		# -----------------------------------------------------------
		# Compute the predictions/accuracy
		# (n x p x k x c) -> (n x p x k)
		# print('23333')
		predictions = tf.argmax(logits, axis=-1, output_type=tf.int32) * top_token_weights

		true_positives = nn.equal(top_label_targets, predictions) * unlabeled_predictions

		correct_label_tokens = nn.equal(top_label_targets, predictions) * top_unlabeled_targets
		# (n x k) -> ()
		n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
		n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
		n_true_positives = tf.reduce_sum(true_positives)
		n_correct_label_tokens = tf.reduce_sum(correct_label_tokens)
		# () - () -> ()
		n_false_positives = n_unlabeled_predictions - n_true_positives
		n_false_negatives = n_unlabeled_targets - n_true_positives
		# (n x k) -> (n)
		n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])

		n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
		n_correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=[1, 2])
		# (n) x 2 -> ()
		n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
		n_correct_label_sequences = tf.reduce_sum(
			nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))

		# -----------------------------------------------------------
		# Populate the output dictionary
		rho = self.loss_interpolation
		outputs['label_predictions'] = predictions
		outputs['label_probabilities'] = label_probabilities
		outputs['label_targets'] = label_targets
		outputs['top_label_targets'] = top_label_targets
		outputs['n_top_label_targets'] = tf.reduce_sum(nn.greater(top_label_targets, 0, dtype=tf.int32))
		outputs['probabilities'] = label_probabilities
		outputs['label_loss'] = label_loss
		# outputs['label_logits'] = transposed_logits*tf.to_float(token_weights)
		# Combination of labeled loss and unlabeled loss
		outputs['loss'] = 2 * ((1 - rho) * outputs['unlabeled_loss'] + rho * label_loss)
		# outputs['loss'] = label_loss * self.loss_rel_interpolation + outputs['loss'] * self.loss_edge_interpolation
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences
		outputs['n_correct_label_tokens'] = n_correct_label_tokens
		outputs['n_correct_label_sequences'] = n_correct_label_sequences

		return outputs

	# =============================================================
	def get_bilinear_classifier_with_args(self, layer, preds, outputs, token_weights, variable_scope=None, reuse=False,
								 debug=False):
		""""""
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear
		top_args_idx = outputs['top_args_idx']

		batch_size, bucket_size, input_size = nn.get_sizes(layer)
		# pdb.set_trace()
		one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
		layer_preds = tf.matmul(one_hot_pred, layer)
		pred_nums = nn.get_sizes(layer_preds)[1]

		# choose pre-difined args according to top_args_idx
		top_args_nums = nn.get_sizes(top_args_idx)[1]


		# layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
		#                         tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)
		#
		# # n x m x m x d -> n x k*d
		# top_layer_args = tf.gather_nd(layer_args, top_args_idx)
		# # n x k x d
		# top_layer_args = tf.reshape(top_layer_args, [batch_size, top_args_nums, input_size * 2])

		if self.span_diff:
			with tf.variable_scope(variable_scope or self.field):
				for i in six.moves.range(0, self.n_layers):
					with tf.variable_scope('LAYERS-FC-%d' % i):
						# n x m x d
						layer = classifiers.hidden(layer, self.hidden_size,
														hidden_func=self.hidden_func,
														hidden_keep_prob=hidden_keep_prob)
		batch_size, bucket_size, input_size = nn.get_sizes(layer)


		args_embed_type = self.args_embed_type

		# n x k x 3 -> n x k x 1
		top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
		top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
		top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])

		args_start_idx = tf.concat([top_args_idx_x, top_args_idx_y], -1)
		args_end_idx = tf.concat([top_args_idx_x, top_args_idx_z], -1)
		# n x k x d
		top_args_start = tf.reshape(tf.gather_nd(layer, args_start_idx), [batch_size, top_args_nums, input_size])
		top_args_end = tf.reshape(tf.gather_nd(layer, args_end_idx), [batch_size, top_args_nums, input_size])
		if args_embed_type == 'diff-sum':
			# n x k x 2d
			top_layer_args = tf.concat([top_args_start + top_args_end, top_args_end-top_args_start], axis=-1)
		elif args_embed_type == 'attention':
			# n x k x 3 -> n x k x 1
			# top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
			top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
			top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])
			attention_score = layer
			with tf.variable_scope(variable_scope or self.field):
				for i in six.moves.range(0, self.n_layers):
					with tf.variable_scope('ATTENTION-FC-%d' % i):
						# n x m x d
						attention_score = classifiers.hidden(attention_score, self.hidden_size,
												   hidden_func=self.hidden_func,
												   hidden_keep_prob=hidden_keep_prob)
				with tf.variable_scope('ATTENTION-TOP'):
						# n x m x 1
						attention_score = classifiers.hidden(attention_score, 1,
												   hidden_func=self.hidden_func,
												   hidden_keep_prob=hidden_keep_prob)

				# n x k x m
				attention_score = tf.transpose(tf.tile(attention_score,[1,1,top_args_nums]),[0,2,1])
				mask = tf.expand_dims(tf.expand_dims(tf.range(bucket_size),0),0)
				mask = tf.cast(tf.tile(mask,[batch_size,top_args_nums,1]),tf.int64)
				mask_tmp1 = tf.where(tf.greater_equal(mask,top_args_idx_y), tf.ones_like(mask), tf.zeros_like(mask))
				mask_tmp2 = tf.where(tf.greater_equal(top_args_idx_z,mask), tf.ones_like(mask), tf.zeros_like(mask))
				mask = tf.where(tf.greater(mask_tmp1+mask_tmp2,1), tf.zeros_like(mask), tf.ones_like(mask))
				# n x k x m  [1,3]->[-inf,0,0,0,-inf,-inf...]
				mask = tf.to_float(mask) * (-1e13)
				# n x k x m [-inf,score,score,score,-inf,-inf...] for every span
				attention_score = mask + attention_score
				attention_weight = tf.math.softmax(attention_score, -1)
				# n x k x m and n x m x d -> n x k x d
				top_layer_args = tf.einsum('nkm,nmd->nkd', attention_weight, layer)
				# concat start and end token embedding
				top_layer_args = tf.concat([top_args_start, top_args_end, top_layer_args], axis=-1)
		elif args_embed_type == 'endpoint':
			# n x k x 2d
			top_layer_args = tf.concat([top_args_start, top_args_end], axis=-1)
		elif args_embed_type == 'max-pooling':
			mask = tf.expand_dims(tf.expand_dims(tf.range(bucket_size), 0), 0)
			mask = tf.cast(tf.tile(mask, [batch_size, top_args_nums, 1]), tf.int64)
			mask_tmp1 = tf.where(tf.greater_equal(mask, top_args_idx_y), tf.ones_like(mask), tf.zeros_like(mask))
			mask_tmp2 = tf.where(tf.greater_equal(top_args_idx_z, mask), tf.ones_like(mask), tf.zeros_like(mask))
			mask = tf.where(tf.greater(mask_tmp1 + mask_tmp2, 1), tf.zeros_like(mask), tf.ones_like(mask))
			# n x k x m  [1,3]->[-inf,0,0,0,-inf,-inf...]
			mask = tf.to_float(mask) * (-1e13)
			# n x k x m [-inf,1,1,1,-inf,-inf...] for every span
			mask += tf.ones_like(mask)
			# n x k x m, n x m x d -> n x k x m x d
			top_layer_args = tf.einsum('nkm,nmd->nkmd', mask, layer)
			# n x k x m x d -> n x k x d
			top_layer_args = tf.reduce_max(top_layer_args, reduction_indices=-2, keep_dims=False)
		elif args_embed_type == 'coherent':
			top_args_start1, top_args_start2, top_args_start3, top_args_start4 = tf.split(top_args_start,tf.constant([560, 560, 40, 40]),-1)
			top_args_end1, top_args_end2, top_args_end3, top_args_end4 = tf.split(top_args_end,tf.constant([560, 560, 40, 40]),-1)
			top_layer_args = tf.concat([top_args_start1, top_args_end2,tf.expand_dims(tf.einsum('nkd,nkd->nk',top_args_start3,top_args_end4),-1)],-1)


		# n x p x m x m
		label_targets = self.placeholder
		# n x m x m x p
		label_targets_trans = tf.transpose(label_targets, [0, 2, 3, 1])
		# n x k*p
		top_label_targets = tf.gather_nd(label_targets_trans, top_args_idx)
		# n x p x k
		top_label_targets = tf.transpose(tf.reshape(top_label_targets, [batch_size, top_args_nums, pred_nums]),
											 [0, 2, 1])

		# n x m x m x p
		token_weights_trans = tf.transpose(token_weights, [0, 2, 3, 1])
		# n x k*p
		top_token_weights = tf.gather_nd(token_weights_trans, top_args_idx)
		# n x p x k
		top_token_weights = tf.transpose(tf.reshape(top_token_weights, [batch_size, top_args_nums, pred_nums]),
										 [0, 2, 1])



		with tf.variable_scope(variable_scope or self.field):

			with tf.variable_scope('Role'):
				layer_role = self.get_role_tensor(reuse=reuse)

			for i in six.moves.range(0, self.n_layers):
				with tf.variable_scope('ARGS-FC-%d' % i):
					top_layer_args = classifiers.hidden(top_layer_args, self.hidden_size,
													hidden_func=self.hidden_func,
													hidden_keep_prob=hidden_keep_prob)

				with tf.variable_scope('PRED-FC-%d' % i):
					layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
													 hidden_func=self.hidden_func,
													 hidden_keep_prob=hidden_keep_prob)
				if self.role_hidden:
					with tf.variable_scope('ROLE-FC-%d' % i):
						layer_role = classifiers.hidden(layer_role, self.hidden_size,
														hidden_func=self.hidden_func,
														hidden_keep_prob=hidden_keep_prob)

			with tf.variable_scope('Classifier'):

				if self.role_tensor:
					# n x p x k x role
					logits = classifiers.span_bilinear_classifier(
						layer_preds, top_layer_args, layer_role,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					# n x p x k x role
					logits = classifiers.diagonal_span_bilinear_classifier(
						layer_preds, top_layer_args, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

		# -----------------------------------------------------------
		# Process the targets

		unlabeled_predictions = outputs['unlabeled_predictions']
		top_unlabeled_targets = outputs['top_unlabeled_targets']
		unlabeled_targets = outputs['unlabeled_targets']

		# -----------------------------------------------------------
		# Compute the probabilities/cross entropy

		# (n x p x k x c) -> (n x p x k x c)
		label_probabilities = tf.nn.softmax(logits) * tf.to_float(tf.expand_dims(top_token_weights, axis=-1))
		# (n x k), (n x k x c), (n x k) -> ()
		label_loss = tf.losses.sparse_softmax_cross_entropy(top_label_targets, logits,
															weights=top_token_weights * top_unlabeled_targets)
		# pdb.set_trace()
		# -----------------------------------------------------------
		# Compute the predictions/accuracy
		# (n x p x k x c) -> (n x p x k)
		# print('23333')
		predictions = tf.argmax(logits, axis=-1, output_type=tf.int32) * top_token_weights

		true_positives = nn.equal(top_label_targets, predictions) * unlabeled_predictions

		correct_label_tokens = nn.equal(top_label_targets, predictions) * top_unlabeled_targets
		# (n x k) -> ()
		n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
		n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
		n_true_positives = tf.reduce_sum(true_positives)
		n_correct_label_tokens = tf.reduce_sum(correct_label_tokens)
		# () - () -> ()
		n_false_positives = n_unlabeled_predictions - n_true_positives
		n_false_negatives = n_unlabeled_targets - n_true_positives
		# (n x k) -> (n)
		n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])

		n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
		n_correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=[1, 2])
		# (n) x 2 -> ()
		n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
		n_correct_label_sequences = tf.reduce_sum(
			nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))

		# -----------------------------------------------------------
		# Populate the output dictionary
		rho = self.loss_interpolation
		outputs['label_predictions'] = predictions
		outputs['label_probabilities'] = label_probabilities
		outputs['label_targets'] = label_targets
		outputs['top_label_targets'] = top_label_targets
		outputs['n_top_label_targets'] = tf.reduce_sum(nn.greater(top_label_targets,0,dtype=tf.int32))
		outputs['probabilities'] = label_probabilities
		outputs['label_loss'] = label_loss
		# outputs['label_logits'] = transposed_logits*tf.to_float(token_weights)
		# Combination of labeled loss and unlabeled loss
		outputs['loss'] = 2 * ((1 - rho) * outputs['unlabeled_loss'] + rho * label_loss)
		# outputs['loss'] = label_loss * self.loss_rel_interpolation + outputs['loss'] * self.loss_edge_interpolation
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences
		outputs['n_correct_label_tokens'] = n_correct_label_tokens
		outputs['n_correct_label_sequences'] = n_correct_label_sequences

		return outputs

	# =============================================================
	def get_bilinear_classifier_with_args_ppred(self, layer, outputs, token_weights, variable_scope=None, reuse=False,
										  debug=False):
		""""""
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear
		top_args_idx = outputs['top_args_idx']

		batch_size, bucket_size, input_size = nn.get_sizes(layer)
		# pdb.set_trace()
		# one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
		layer_preds = layer
		pred_nums = nn.get_sizes(layer_preds)[1]

		# choose pre-difined args according to top_args_idx
		top_args_nums = nn.get_sizes(top_args_idx)[1]

		# layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
		#                         tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)
		#
		# # n x m x m x d -> n x k*d
		# top_layer_args = tf.gather_nd(layer_args, top_args_idx)
		# # n x k x d
		# top_layer_args = tf.reshape(top_layer_args, [batch_size, top_args_nums, input_size * 2])

		if self.span_diff:
			with tf.variable_scope(variable_scope or self.field):
				for i in six.moves.range(0, self.n_layers):
					with tf.variable_scope('LAYERS-FC-%d' % i):
						# n x m x d
						layer = classifiers.hidden(layer, self.hidden_size,
												   hidden_func=self.hidden_func,
												   hidden_keep_prob=hidden_keep_prob)
		batch_size, bucket_size, input_size = nn.get_sizes(layer)

		args_embed_type = self.args_embed_type

		# n x k x 3 -> n x k x 1
		top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
		top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
		top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])

		args_start_idx = tf.concat([top_args_idx_x, top_args_idx_y], -1)
		args_end_idx = tf.concat([top_args_idx_x, top_args_idx_z], -1)
		# n x k x d
		top_args_start = tf.reshape(tf.gather_nd(layer, args_start_idx), [batch_size, top_args_nums, input_size])
		top_args_end = tf.reshape(tf.gather_nd(layer, args_end_idx), [batch_size, top_args_nums, input_size])
		if args_embed_type == 'diff-sum':
			# n x k x 2d
			top_layer_args = tf.concat([top_args_start + top_args_end, top_args_end - top_args_start], axis=-1)
		elif args_embed_type == 'attention':
			# n x k x 3 -> n x k x 1
			# top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
			top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
			top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])
			attention_score = layer
			with tf.variable_scope(variable_scope or self.field):
				for i in six.moves.range(0, self.n_layers):
					with tf.variable_scope('ATTENTION-FC-%d' % i):
						# n x m x d
						attention_score = classifiers.hidden(attention_score, self.hidden_size,
															 hidden_func=self.hidden_func,
															 hidden_keep_prob=hidden_keep_prob)
				with tf.variable_scope('ATTENTION-TOP'):
					# n x m x 1
					attention_score = classifiers.hidden(attention_score, 1,
														 hidden_func=self.hidden_func,
														 hidden_keep_prob=hidden_keep_prob)

				# n x k x m
				attention_score = tf.transpose(tf.tile(attention_score, [1, 1, top_args_nums]), [0, 2, 1])
				mask = tf.expand_dims(tf.expand_dims(tf.range(bucket_size), 0), 0)
				mask = tf.cast(tf.tile(mask, [batch_size, top_args_nums, 1]), tf.int64)
				mask_tmp1 = tf.where(tf.greater_equal(mask, top_args_idx_y), tf.ones_like(mask), tf.zeros_like(mask))
				mask_tmp2 = tf.where(tf.greater_equal(top_args_idx_z, mask), tf.ones_like(mask), tf.zeros_like(mask))
				mask = tf.where(tf.greater(mask_tmp1 + mask_tmp2, 1), tf.zeros_like(mask), tf.ones_like(mask))
				# n x k x m  [1,3]->[-inf,0,0,0,-inf,-inf...]
				mask = tf.to_float(mask) * (-1e13)
				# n x k x m [-inf,score,score,score,-inf,-inf...] for every span
				attention_score = mask + attention_score
				attention_weight = tf.math.softmax(attention_score, -1)
				# n x k x m and n x m x d -> n x k x d
				top_layer_args = tf.einsum('nkm,nmd->nkd', attention_weight, layer)
				# concat start and end token embedding
				top_layer_args = tf.concat([top_args_start, top_args_end, top_layer_args], axis=-1)
		elif args_embed_type == 'endpoint':
			# n x k x 2d
			top_layer_args = tf.concat([top_args_start, top_args_end], axis=-1)
		elif args_embed_type == 'max-pooling':
			mask = tf.expand_dims(tf.expand_dims(tf.range(bucket_size), 0), 0)
			mask = tf.cast(tf.tile(mask, [batch_size, top_args_nums, 1]), tf.int64)
			mask_tmp1 = tf.where(tf.greater_equal(mask, top_args_idx_y), tf.ones_like(mask), tf.zeros_like(mask))
			mask_tmp2 = tf.where(tf.greater_equal(top_args_idx_z, mask), tf.ones_like(mask), tf.zeros_like(mask))
			mask = tf.where(tf.greater(mask_tmp1 + mask_tmp2, 1), tf.zeros_like(mask), tf.ones_like(mask))
			# n x k x m  [1,3]->[-inf,0,0,0,-inf,-inf...]
			mask = tf.to_float(mask) * (-1e13)
			# n x k x m [-inf,1,1,1,-inf,-inf...] for every span
			mask += tf.ones_like(mask)
			# n x k x m, n x m x d -> n x k x m x d
			top_layer_args = tf.einsum('nkm,nmd->nkmd', mask, layer)
			# n x k x m x d -> n x k x d
			top_layer_args = tf.reduce_max(top_layer_args, reduction_indices=-2, keep_dims=False)
		elif args_embed_type == 'coherent':
			top_args_start1, top_args_start2, top_args_start3, top_args_start4 = tf.split(top_args_start, tf.constant(
				[560, 560, 40, 40]), -1)
			top_args_end1, top_args_end2, top_args_end3, top_args_end4 = tf.split(top_args_end,
																				  tf.constant([560, 560, 40, 40]), -1)
			top_layer_args = tf.concat([top_args_start1, top_args_end2,
										tf.expand_dims(tf.einsum('nkd,nkd->nk', top_args_start3, top_args_end4), -1)],
									   -1)

		# n x p x m x m
		label_targets = self.placeholder
		# n x m x m x p
		label_targets_trans = tf.transpose(label_targets, [0, 2, 3, 1])
		# n x k*p
		top_label_targets = tf.gather_nd(label_targets_trans, top_args_idx)
		# n x p x k
		top_label_targets = tf.transpose(tf.reshape(top_label_targets, [batch_size, top_args_nums, pred_nums]),
										 [0, 2, 1])

		# n x m x m x p
		token_weights_trans = tf.transpose(token_weights, [0, 2, 3, 1])
		# n x k*p
		top_token_weights = tf.gather_nd(token_weights_trans, top_args_idx)
		# n x p x k
		top_token_weights = tf.transpose(tf.reshape(top_token_weights, [batch_size, top_args_nums, pred_nums]),
										 [0, 2, 1])

		with tf.variable_scope(variable_scope or self.field):

			with tf.variable_scope('Role'):
				layer_role = self.get_role_tensor(reuse=reuse)

			for i in six.moves.range(0, self.n_layers):
				with tf.variable_scope('ARGS-FC-%d' % i):
					top_layer_args = classifiers.hidden(top_layer_args, self.hidden_size,
														hidden_func=self.hidden_func,
														hidden_keep_prob=hidden_keep_prob)

				with tf.variable_scope('PRED-FC-%d' % i):
					layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
													 hidden_func=self.hidden_func,
													 hidden_keep_prob=hidden_keep_prob)
				if self.role_hidden:
					with tf.variable_scope('ROLE-FC-%d' % i):
						layer_role = classifiers.hidden(layer_role, self.hidden_size,
														hidden_func=self.hidden_func,
														hidden_keep_prob=hidden_keep_prob)

			with tf.variable_scope('Classifier'):

				if self.role_tensor:
					# n x p x k x role
					logits = classifiers.span_bilinear_classifier(
						layer_preds, top_layer_args, layer_role,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					# n x p x k x role
					logits = classifiers.diagonal_span_bilinear_classifier(
						layer_preds, top_layer_args, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

		# -----------------------------------------------------------
		# Process the targets

		unlabeled_predictions = outputs['unlabeled_predictions']
		top_unlabeled_targets = outputs['top_unlabeled_targets']
		unlabeled_targets = outputs['unlabeled_targets']

		# -----------------------------------------------------------
		# Compute the probabilities/cross entropy

		# (n x p x k x c) -> (n x p x k x c)
		label_probabilities = tf.nn.softmax(logits) * tf.to_float(tf.expand_dims(top_token_weights, axis=-1))
		# (n x k), (n x k x c), (n x k) -> ()
		label_loss = tf.losses.sparse_softmax_cross_entropy(top_label_targets, logits,
															weights=top_token_weights * top_unlabeled_targets)
		# pdb.set_trace()
		# -----------------------------------------------------------
		# Compute the predictions/accuracy
		# (n x p x k x c) -> (n x p x k)
		# print('23333')
		predictions = tf.argmax(logits, axis=-1, output_type=tf.int32) * top_token_weights

		true_positives = nn.equal(top_label_targets, predictions) * unlabeled_predictions

		correct_label_tokens = nn.equal(top_label_targets, predictions) * top_unlabeled_targets
		# (n x k) -> ()
		n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
		n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
		n_true_positives = tf.reduce_sum(true_positives)
		n_correct_label_tokens = tf.reduce_sum(correct_label_tokens)
		# () - () -> ()
		n_false_positives = n_unlabeled_predictions - n_true_positives
		n_false_negatives = n_unlabeled_targets - n_true_positives
		# (n x k) -> (n)
		n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])

		n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
		n_correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=[1, 2])
		# (n) x 2 -> ()
		n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
		n_correct_label_sequences = tf.reduce_sum(
			nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))

		# -----------------------------------------------------------
		# Populate the output dictionary
		rho = self.loss_interpolation
		outputs['label_predictions'] = predictions
		outputs['label_probabilities'] = label_probabilities
		outputs['label_targets'] = label_targets
		outputs['top_label_targets'] = top_label_targets
		outputs['n_top_label_targets'] = tf.reduce_sum(nn.greater(top_label_targets, 0, dtype=tf.int32))
		outputs['probabilities'] = label_probabilities
		outputs['label_loss'] = label_loss
		# outputs['label_logits'] = transposed_logits*tf.to_float(token_weights)
		# Combination of labeled loss and unlabeled loss
		outputs['loss'] = 2 * ((1 - rho) * outputs['unlabeled_loss'] + rho * label_loss)
		# outputs['loss'] = label_loss * self.loss_rel_interpolation + outputs['loss'] * self.loss_edge_interpolation
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences
		outputs['n_correct_label_tokens'] = n_correct_label_tokens
		outputs['n_correct_label_sequences'] = n_correct_label_sequences

		return outputs

	# =============================================================
	def get_bilinear_classifier_with_args_syntax_ppred(self, layer, outputs, syntax_indicator_vocab, syntax_label_vocab, token_weights, variable_scope=None,
												reuse=False,
												debug=False):
			""""""
			hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
			add_linear = self.add_linear
			top_args_idx = outputs['top_args_idx']

			batch_size, bucket_size, input_size = nn.get_sizes(layer)
			# pdb.set_trace()
			# one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
			layer_preds = layer
			pred_nums = nn.get_sizes(layer_preds)[1]

			# choose pre-difined args according to top_args_idx
			top_args_nums = nn.get_sizes(top_args_idx)[1]

			# layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
			#                         tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)
			#
			# # n x m x m x d -> n x k*d
			# top_layer_args = tf.gather_nd(layer_args, top_args_idx)
			# # n x k x d
			# top_layer_args = tf.reshape(top_layer_args, [batch_size, top_args_nums, input_size * 2])

			if self.span_diff:
				with tf.variable_scope(variable_scope or self.field):
					for i in six.moves.range(0, self.n_layers):
						with tf.variable_scope('LAYERS-FC-%d' % i):
							# n x m x d
							layer = classifiers.hidden(layer, self.hidden_size,
													   hidden_func=self.hidden_func,
													   hidden_keep_prob=hidden_keep_prob)
			batch_size, bucket_size, input_size = nn.get_sizes(layer)

			args_embed_type = self.args_embed_type

			# --------------------use syntax information, constiuents label----------------------
			syntax_indicator = syntax_indicator_vocab.placeholder
			syntax_indicator = tf.transpose(syntax_indicator, [0, 2, 1])

			syntax_label = syntax_label_vocab.placeholder
			syntax_label = tf.transpose(syntax_label, [0, 2, 1])

			# n x k
			top_span_syntax_indicator = tf.gather_nd(syntax_indicator, top_args_idx)
			top_span_syntax_label = tf.gather_nd(syntax_label, top_args_idx)

			# ----------get syntax indicator embedding-------------------------------------
			# embed_keep_prob = 1 if reuse else (embed_keep_prob or self.embed_keep_prob)

			# n x k x 100
			with tf.variable_scope("RelSyntaxIndicator"):
				top_span_syntax_indicator_embed = embeddings.token_embedding_lookup(2, 50,
																					top_span_syntax_indicator,
																					nonzero_init=True,
																					reuse=reuse)

			with tf.variable_scope("RelSyntaxLabel"):
				top_span_syntax_label_embed = embeddings.token_embedding_lookup(len(syntax_label_vocab), 50,
																				top_span_syntax_label,
																				nonzero_init=True,
																				reuse=reuse)



			# n x k x 3 -> n x k x 1
			top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
			top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
			top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])

			args_start_idx = tf.concat([top_args_idx_x, top_args_idx_y], -1)
			args_end_idx = tf.concat([top_args_idx_x, top_args_idx_z], -1)
			# n x k x d
			top_args_start = tf.reshape(tf.gather_nd(layer, args_start_idx), [batch_size, top_args_nums, input_size])
			top_args_end = tf.reshape(tf.gather_nd(layer, args_end_idx), [batch_size, top_args_nums, input_size])
			if args_embed_type == 'diff-sum':
				# n x k x 2d
				top_layer_args = tf.concat([top_args_start + top_args_end, top_args_end - top_args_start], axis=-1)
			elif args_embed_type == 'attention':
				# n x k x 3 -> n x k x 1
				# top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
				top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
				top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])
				attention_score = layer
				with tf.variable_scope(variable_scope or self.field):
					for i in six.moves.range(0, self.n_layers):
						with tf.variable_scope('ATTENTION-FC-%d' % i):
							# n x m x d
							attention_score = classifiers.hidden(attention_score, self.hidden_size,
																 hidden_func=self.hidden_func,
																 hidden_keep_prob=hidden_keep_prob)
					with tf.variable_scope('ATTENTION-TOP'):
						# n x m x 1
						attention_score = classifiers.hidden(attention_score, 1,
															 hidden_func=self.hidden_func,
															 hidden_keep_prob=hidden_keep_prob)

					# n x k x m
					attention_score = tf.transpose(tf.tile(attention_score, [1, 1, top_args_nums]), [0, 2, 1])
					mask = tf.expand_dims(tf.expand_dims(tf.range(bucket_size), 0), 0)
					mask = tf.cast(tf.tile(mask, [batch_size, top_args_nums, 1]), tf.int64)
					mask_tmp1 = tf.where(tf.greater_equal(mask, top_args_idx_y), tf.ones_like(mask),
										 tf.zeros_like(mask))
					mask_tmp2 = tf.where(tf.greater_equal(top_args_idx_z, mask), tf.ones_like(mask),
										 tf.zeros_like(mask))
					mask = tf.where(tf.greater(mask_tmp1 + mask_tmp2, 1), tf.zeros_like(mask), tf.ones_like(mask))
					# n x k x m  [1,3]->[-inf,0,0,0,-inf,-inf...]
					mask = tf.to_float(mask) * (-1e13)
					# n x k x m [-inf,score,score,score,-inf,-inf...] for every span
					attention_score = mask + attention_score
					attention_weight = tf.math.softmax(attention_score, -1)
					# n x k x m and n x m x d -> n x k x d
					top_layer_args = tf.einsum('nkm,nmd->nkd', attention_weight, layer)
					# concat start and end token embedding
					top_layer_args = tf.concat([top_args_start, top_args_end, top_layer_args], axis=-1)
			elif args_embed_type == 'endpoint':
				# n x k x 2d
				top_layer_args = tf.concat([top_args_start, top_args_end], axis=-1)
			elif args_embed_type == 'max-pooling':
				mask = tf.expand_dims(tf.expand_dims(tf.range(bucket_size), 0), 0)
				mask = tf.cast(tf.tile(mask, [batch_size, top_args_nums, 1]), tf.int64)
				mask_tmp1 = tf.where(tf.greater_equal(mask, top_args_idx_y), tf.ones_like(mask), tf.zeros_like(mask))
				mask_tmp2 = tf.where(tf.greater_equal(top_args_idx_z, mask), tf.ones_like(mask), tf.zeros_like(mask))
				mask = tf.where(tf.greater(mask_tmp1 + mask_tmp2, 1), tf.zeros_like(mask), tf.ones_like(mask))
				# n x k x m  [1,3]->[-inf,0,0,0,-inf,-inf...]
				mask = tf.to_float(mask) * (-1e13)
				# n x k x m [-inf,1,1,1,-inf,-inf...] for every span
				mask += tf.ones_like(mask)
				# n x k x m, n x m x d -> n x k x m x d
				top_layer_args = tf.einsum('nkm,nmd->nkmd', mask, layer)
				# n x k x m x d -> n x k x d
				top_layer_args = tf.reduce_max(top_layer_args, reduction_indices=-2, keep_dims=False)
			elif args_embed_type == 'coherent':
				top_args_start1, top_args_start2, top_args_start3, top_args_start4 = tf.split(top_args_start,
																							  tf.constant(
																								  [560, 560, 40, 40]),
																							  -1)
				top_args_end1, top_args_end2, top_args_end3, top_args_end4 = tf.split(top_args_end,
																					  tf.constant([560, 560, 40, 40]),
																					  -1)
				top_layer_args = tf.concat([top_args_start1, top_args_end2,
											tf.expand_dims(tf.einsum('nkd,nkd->nk', top_args_start3, top_args_end4),
														   -1)],-1)

				# -------------------------concat syntax indicator embedding----------------------------------------
				top_layer_args = tf.concat(
					[top_layer_args, top_span_syntax_indicator_embed, top_span_syntax_label_embed], -1)


			# n x p x m x m
			label_targets = self.placeholder
			# n x m x m x p
			label_targets_trans = tf.transpose(label_targets, [0, 2, 3, 1])
			# n x k*p
			top_label_targets = tf.gather_nd(label_targets_trans, top_args_idx)
			# n x p x k
			top_label_targets = tf.transpose(tf.reshape(top_label_targets, [batch_size, top_args_nums, pred_nums]),
											 [0, 2, 1])

			# n x m x m x p
			token_weights_trans = tf.transpose(token_weights, [0, 2, 3, 1])
			# n x k*p
			top_token_weights = tf.gather_nd(token_weights_trans, top_args_idx)
			# n x p x k
			top_token_weights = tf.transpose(tf.reshape(top_token_weights, [batch_size, top_args_nums, pred_nums]),
											 [0, 2, 1])

			with tf.variable_scope(variable_scope or self.field):

				with tf.variable_scope('Role'):
					layer_role = self.get_role_tensor(reuse=reuse)

				for i in six.moves.range(0, self.n_layers):
					with tf.variable_scope('ARGS-FC-%d' % i):
						top_layer_args = classifiers.hidden(top_layer_args, self.hidden_size,
															hidden_func=self.hidden_func,
															hidden_keep_prob=hidden_keep_prob)

					with tf.variable_scope('PRED-FC-%d' % i):
						layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
														 hidden_func=self.hidden_func,
														 hidden_keep_prob=hidden_keep_prob)
					if self.role_hidden:
						with tf.variable_scope('ROLE-FC-%d' % i):
							layer_role = classifiers.hidden(layer_role, self.hidden_size,
															hidden_func=self.hidden_func,
															hidden_keep_prob=hidden_keep_prob)

				with tf.variable_scope('Classifier'):

					if self.role_tensor:
						# n x p x k x role
						logits = classifiers.span_bilinear_classifier(
							layer_preds, top_layer_args, layer_role,
							hidden_keep_prob=hidden_keep_prob,
							add_linear=add_linear)
					else:
						# n x p x k x role
						logits = classifiers.diagonal_span_bilinear_classifier(
							layer_preds, top_layer_args, len(self),
							hidden_keep_prob=hidden_keep_prob,
							add_linear=add_linear)

			# -----------------------------------------------------------
			# Process the targets

			unlabeled_predictions = outputs['unlabeled_predictions']
			top_unlabeled_targets = outputs['top_unlabeled_targets']
			unlabeled_targets = outputs['unlabeled_targets']

			# -----------------------------------------------------------
			# Compute the probabilities/cross entropy

			# (n x p x k x c) -> (n x p x k x c)
			label_probabilities = tf.nn.softmax(logits) * tf.to_float(tf.expand_dims(top_token_weights, axis=-1))
			# (n x k), (n x k x c), (n x k) -> ()
			label_loss = tf.losses.sparse_softmax_cross_entropy(top_label_targets, logits,
																weights=top_token_weights * top_unlabeled_targets)
			# pdb.set_trace()
			# -----------------------------------------------------------
			# Compute the predictions/accuracy
			# (n x p x k x c) -> (n x p x k)
			# print('23333')
			predictions = tf.argmax(logits, axis=-1, output_type=tf.int32) * top_token_weights

			true_positives = nn.equal(top_label_targets, predictions) * unlabeled_predictions

			correct_label_tokens = nn.equal(top_label_targets, predictions) * top_unlabeled_targets
			# (n x k) -> ()
			n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
			n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
			n_true_positives = tf.reduce_sum(true_positives)
			n_correct_label_tokens = tf.reduce_sum(correct_label_tokens)
			# () - () -> ()
			n_false_positives = n_unlabeled_predictions - n_true_positives
			n_false_negatives = n_unlabeled_targets - n_true_positives
			# (n x k) -> (n)
			n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])

			n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
			n_correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=[1, 2])
			# (n) x 2 -> ()
			n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
			n_correct_label_sequences = tf.reduce_sum(
				nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))

			# -----------------------------------------------------------
			# Populate the output dictionary
			rho = self.loss_interpolation
			outputs['label_predictions'] = predictions
			outputs['label_probabilities'] = label_probabilities
			outputs['label_targets'] = label_targets
			outputs['top_label_targets'] = top_label_targets
			outputs['n_top_label_targets'] = tf.reduce_sum(nn.greater(top_label_targets, 0, dtype=tf.int32))
			outputs['probabilities'] = label_probabilities
			outputs['label_loss'] = label_loss
			# outputs['label_logits'] = transposed_logits*tf.to_float(token_weights)
			# Combination of labeled loss and unlabeled loss
			outputs['loss'] = 2 * ((1 - rho) * outputs['unlabeled_loss'] + rho * label_loss)
			# outputs['loss'] = label_loss * self.loss_rel_interpolation + outputs['loss'] * self.loss_edge_interpolation
			outputs['n_true_positives'] = n_true_positives
			outputs['n_false_positives'] = n_false_positives
			outputs['n_false_negatives'] = n_false_negatives
			outputs['n_correct_sequences'] = n_correct_sequences
			outputs['n_correct_label_tokens'] = n_correct_label_tokens
			outputs['n_correct_label_sequences'] = n_correct_label_sequences

			return outputs



	# =============================================================
	def get_unfactored_bilinear_classifier_with_args(self, layer, preds, token_weights, top_args_idx, variable_scope=None,
										  reuse=False, debug=False):
			""""""
			outputs = {}
			hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
			add_linear = self.add_linear

			batch_size, bucket_size, input_size = nn.get_sizes(layer)
			# pdb.set_trace()
			one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
			layer_preds = tf.matmul(one_hot_pred, layer)
			pred_nums = nn.get_sizes(layer_preds)[1]
			# choose pre-difined args according to top_args_idx
			top_args_nums = nn.get_sizes(top_args_idx)[1]

			# with tf.variable_scope(variable_scope or self.field):
			#
			# 	with tf.variable_scope('Role'):
			# 		layer_role = self.get_role_tensor(reuse=reuse)
			#
			# 	for i in six.moves.range(0, self.n_layers):
			# 		with tf.variable_scope('ARGS-FC-%d' % i):
			# 			layer = classifiers.hidden(layer, self.hidden_size,
			# 			                                    hidden_func=self.hidden_func,
			# 			                                    hidden_keep_prob=hidden_keep_prob)
			#
			# 		with tf.variable_scope('PRED-FC-%d' % i):
			# 			layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
			# 			                                 hidden_func=self.hidden_func,
			# 			                                 hidden_keep_prob=hidden_keep_prob)
			# 		if self.role_hidden:
			# 			with tf.variable_scope('ROLE-FC-%d' % i):
			# 				layer_role = classifiers.hidden(layer_role, self.hidden_size,
			# 				                                hidden_func=self.hidden_func,
			# 				                                hidden_keep_prob=hidden_keep_prob)


			# layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
			#                         tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)
			#
			# # n x m x m x d -> n x k*d
			# top_layer_args = tf.gather_nd(layer_args, top_args_idx)
			# # n x k x d
			# top_layer_args = tf.reshape(top_layer_args, [batch_size, top_args_nums, input_size * 2])

			batch_size, bucket_size, input_size = nn.get_sizes(layer)

			args_embed_type = self.args_embed_type

			# n x k x 3 -> n x k x 1
			top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
			top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
			top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])

			args_start_idx = tf.concat([top_args_idx_x, top_args_idx_y], -1)
			args_end_idx = tf.concat([top_args_idx_x, top_args_idx_z], -1)
			# n x k x d
			top_args_start = tf.reshape(tf.gather_nd(layer, args_start_idx),
										[batch_size, top_args_nums, input_size])
			top_args_end = tf.reshape(tf.gather_nd(layer, args_end_idx), [batch_size, top_args_nums, input_size])


			if args_embed_type == 's+e_s-e_c':
				# n x k x 2d
				top_layer_args = tf.concat([top_args_start + top_args_end, top_args_end - top_args_start], axis=-1)
			elif args_embed_type == 'attention':
				# n x k x 3 -> n x k x 1
				# top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
				top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
				top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])
				attention_score = layer
				with tf.variable_scope(variable_scope or self.field):
					for i in six.moves.range(0, self.n_layers):
						with tf.variable_scope('ATTENTION-FC-%d' % i):
							# n x m x d
							attention_score = classifiers.hidden(attention_score, self.hidden_size,
																 hidden_func=self.hidden_func,
																 hidden_keep_prob=hidden_keep_prob)
					with tf.variable_scope('ATTENTION-TOP'):
						# n x m x 1
						attention_score = classifiers.hidden(attention_score, 1,
															 hidden_func=self.hidden_func,
															 hidden_keep_prob=hidden_keep_prob)

					# n x k x m
					attention_score = tf.transpose(tf.tile(attention_score, [1, 1, top_args_nums]), [0, 2, 1])
					mask = tf.expand_dims(tf.expand_dims(tf.range(bucket_size), 0), 0)
					mask = tf.cast(tf.tile(mask, [batch_size, top_args_nums, 1]), tf.int64)
					mask_tmp1 = tf.where(tf.greater_equal(mask, top_args_idx_y), tf.ones_like(mask),
										 tf.zeros_like(mask))
					mask_tmp2 = tf.where(tf.greater_equal(top_args_idx_z, mask), tf.ones_like(mask),
										 tf.zeros_like(mask))
					mask = tf.where(tf.greater(mask_tmp1 + mask_tmp2, 1), tf.zeros_like(mask), tf.ones_like(mask))
					# n x k x m  [1,3]->[-inf,0,0,0,-inf,-inf...]
					mask = tf.to_float(mask) * (-1e13)
					# n x k x m [-inf,score,score,score,-inf,-inf...] for every span
					attention_score = mask + attention_score
					attention_weight = tf.math.softmax(attention_score, -1)
					# n x k x m and n x m x d -> n x k x d
					top_layer_args = tf.einsum('nkm,nmd->nkd', attention_weight, layer)
					# concat start and end token embedding
					top_layer_args = tf.concat([top_args_start, top_args_end, top_layer_args], axis=-1)
			# n x p x m x m
			label_targets = self.placeholder
			# n x p x m x m
			unlabeled_targets = nn.greater(label_targets, 0)

			# n x m x m x p
			label_targets_trans = tf.transpose(label_targets, [0, 2, 3, 1])
			# n x k*p
			top_label_targets = tf.gather_nd(label_targets_trans, top_args_idx)
			# n x p x k
			top_label_targets = tf.transpose(tf.reshape(top_label_targets, [batch_size, top_args_nums, pred_nums]),
											 [0, 2, 1])

			# n x p x k
			top_unlabeled_targets = nn.greater(top_label_targets, 0)

			# n x m x m x p
			token_weights_trans = tf.transpose(token_weights, [0, 2, 3, 1])
			# n x k*p
			top_token_weights = tf.gather_nd(token_weights_trans, top_args_idx)
			# n x p x k
			top_token_weights = tf.transpose(tf.reshape(top_token_weights, [batch_size, top_args_nums, pred_nums]),
											 [0, 2, 1])

			with tf.variable_scope(variable_scope or self.field):

				with tf.variable_scope('Role'):
					layer_role = self.get_role_tensor(reuse=reuse)

				for i in six.moves.range(0, self.n_layers):
					with tf.variable_scope('ARGS-FC-%d' % i):
						top_layer_args = classifiers.hidden(top_layer_args, self.hidden_size,
															hidden_func=self.hidden_func,
															hidden_keep_prob=hidden_keep_prob)

					with tf.variable_scope('PRED-FC-%d' % i):
						layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
														 hidden_func=self.hidden_func,
														 hidden_keep_prob=hidden_keep_prob)
					if self.role_hidden:
						with tf.variable_scope('ROLE-FC-%d' % i):
							layer_role = classifiers.hidden(layer_role, self.hidden_size,
															hidden_func=self.hidden_func,
															hidden_keep_prob=hidden_keep_prob)

				with tf.variable_scope('Classifier'):

					if self.role_tensor:
						# n x p x k x role
						logits = classifiers.span_bilinear_classifier(
							layer_preds, top_layer_args, layer_role,
							hidden_keep_prob=hidden_keep_prob,
							add_linear=add_linear)
					else:
						# n x p x k x role
						logits = classifiers.diagonal_span_bilinear_classifier(
							layer_preds, top_layer_args, len(self),
							hidden_keep_prob=hidden_keep_prob,
							add_linear=add_linear)



			# -----------------------------------------------------------
			# Compute the probabilities/cross entropy

			# (n x p x k x c) -> (n x p x k x c)
			label_probabilities = tf.nn.softmax(logits) * tf.to_float(tf.expand_dims(top_token_weights, axis=-1))
			# (n x k), (n x k x c), (n x k) -> ()
			label_loss = tf.losses.sparse_softmax_cross_entropy(top_label_targets, logits,
																weights=top_token_weights)
			# pdb.set_trace()
			# -----------------------------------------------------------
			# Compute the predictions/accuracy
			# (n x p x k x c) -> (n x p x k)
			# print('23333')
			predictions = tf.argmax(logits, axis=-1, output_type=tf.int32) * top_token_weights
			# (n x p x k) -> (n x p x k)
			unlabeled_predictions = nn.greater(predictions, 0)
			# (n x p x k) (*) (n x p x k) -> (n x p x k)
			unlabeled_true_positives = unlabeled_predictions * top_unlabeled_targets
			true_positives = nn.equal(top_label_targets, predictions) * top_unlabeled_targets
			# (n x p x k) -> ()
			n_predictions = tf.reduce_sum(unlabeled_predictions)
			# (n x p x m x m) -> ()
			n_targets = tf.reduce_sum(unlabeled_targets)
			n_unlabeled_true_positives = tf.reduce_sum(unlabeled_true_positives)
			n_true_positives = tf.reduce_sum(true_positives)
			# () - () -> ()
			n_unlabeled_false_positives = n_predictions - n_unlabeled_true_positives
			n_unlabeled_false_negatives = n_targets - n_unlabeled_true_positives
			n_false_positives = n_predictions - n_true_positives
			n_false_negatives = n_targets - n_true_positives
			# (n x p x m x m) -> (n)
			n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])
			n_unlabeled_true_positives_per_sequence = tf.reduce_sum(unlabeled_true_positives, axis=[1, 2])
			n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
			# (n) x 2 -> ()
			n_correct_unlabeled_sequences = tf.reduce_sum(
				nn.equal(n_unlabeled_true_positives_per_sequence, n_targets_per_sequence))
			n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

			outputs['unlabeled_targets'] = unlabeled_targets
			outputs['top_unlabeled_targets'] = top_unlabeled_targets
			outputs['label_targets'] = label_targets
			outputs['top_label_targets'] = top_label_targets

			outputs['probabilities'] = label_probabilities
			outputs['logits'] = logits

			outputs['loss'] = label_loss
			outputs['unlabeled_loss'] = label_loss

			outputs['unlabeled_predictions'] = unlabeled_predictions
			outputs['label_predictions'] = predictions

			outputs['n_unlabeled_true_positives'] = n_unlabeled_true_positives
			outputs['n_unlabeled_false_positives'] = n_unlabeled_false_positives
			outputs['n_unlabeled_false_negatives'] = n_unlabeled_false_negatives
			outputs['n_correct_unlabeled_sequences'] = n_correct_unlabeled_sequences
			outputs['n_true_positives'] = n_true_positives
			outputs['n_false_positives'] = n_false_positives
			outputs['n_false_negatives'] = n_false_negatives
			outputs['n_correct_sequences'] = n_correct_sequences

			outputs['top_args_idx'] = top_args_idx

			return outputs

	# =============================================================
	def get_unfactored_bilinear_classifier_with_args_uni(self, layer, preds, token_weights, top_args_idx,
													 variable_scope=None,
													 reuse=False, debug=False):
		""""""
		outputs = {}
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear

		batch_size, bucket_size, input_size = nn.get_sizes(layer)
		# pdb.set_trace()
		one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
		layer_preds = tf.matmul(one_hot_pred, layer)
		pred_nums = nn.get_sizes(layer_preds)[1]
		# choose pre-difined args according to top_args_idx
		top_args_nums = nn.get_sizes(top_args_idx)[1]

		# with tf.variable_scope(variable_scope or self.field):
		#
		# 	with tf.variable_scope('Role'):
		# 		layer_role = self.get_role_tensor(reuse=reuse)
		#
		# 	for i in six.moves.range(0, self.n_layers):
		# 		with tf.variable_scope('ARGS-FC-%d' % i):
		# 			layer = classifiers.hidden(layer, self.hidden_size,
		# 			                           hidden_func=self.hidden_func,
		# 			                           hidden_keep_prob=hidden_keep_prob)
		#
		# 		with tf.variable_scope('PRED-FC-%d' % i):
		# 			layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
		# 			                                 hidden_func=self.hidden_func,
		# 			                                 hidden_keep_prob=hidden_keep_prob)
		# 		if self.role_hidden:
		# 			with tf.variable_scope('ROLE-FC-%d' % i):
		# 				layer_role = classifiers.hidden(layer_role, self.hidden_size,
		# 				                                hidden_func=self.hidden_func,
		# 				                                hidden_keep_prob=hidden_keep_prob)

		# layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
		#                         tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)
		#
		# # n x m x m x d -> n x k*d
		# top_layer_args = tf.gather_nd(layer_args, top_args_idx)
		# # n x k x d
		# top_layer_args = tf.reshape(top_layer_args, [batch_size, top_args_nums, input_size * 2])

		batch_size, bucket_size, input_size = nn.get_sizes(layer)

		args_embed_type = self.args_embed_type

		# n x k x 3 -> n x k x 1
		top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
		top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
		top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])

		args_start_idx = tf.concat([top_args_idx_x, top_args_idx_y], -1)
		args_end_idx = tf.concat([top_args_idx_x, top_args_idx_z], -1)
		# n x k x d
		top_args_start = tf.reshape(tf.gather_nd(layer, args_start_idx),
									[batch_size, top_args_nums, input_size])
		top_args_end = tf.reshape(tf.gather_nd(layer, args_end_idx), [batch_size, top_args_nums, input_size])

		if args_embed_type == 'diff-sum':
			# n x k x 2d
			top_layer_args = tf.concat([top_args_start + top_args_end, top_args_end - top_args_start], axis=-1)
		elif args_embed_type == 'attention':
			# n x k x 3 -> n x k x 1
			# top_args_idx_x = tf.slice(top_args_idx, [0, 0, 0], [-1, -1, 1])
			top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
			top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])
			attention_score = layer
			with tf.variable_scope(variable_scope or self.field):
				for i in six.moves.range(0, self.n_layers):
					with tf.variable_scope('ATTENTION-FC-%d' % i):
						# n x m x d
						attention_score = classifiers.hidden(attention_score, self.hidden_size,
															 hidden_func=self.hidden_func,
															 hidden_keep_prob=hidden_keep_prob)
				with tf.variable_scope('ATTENTION-TOP'):
					# n x m x 1
					attention_score = classifiers.hidden(attention_score, 1,
														 hidden_func=self.hidden_func,
														 hidden_keep_prob=hidden_keep_prob)

				# n x k x m
				attention_score = tf.transpose(tf.tile(attention_score, [1, 1, top_args_nums]), [0, 2, 1])
				mask = tf.expand_dims(tf.expand_dims(tf.range(bucket_size), 0), 0)
				mask = tf.cast(tf.tile(mask, [batch_size, top_args_nums, 1]), tf.int64)
				mask_tmp1 = tf.where(tf.greater_equal(mask, top_args_idx_y), tf.ones_like(mask),
									 tf.zeros_like(mask))
				mask_tmp2 = tf.where(tf.greater_equal(top_args_idx_z, mask), tf.ones_like(mask),
									 tf.zeros_like(mask))
				mask = tf.where(tf.greater(mask_tmp1 + mask_tmp2, 1), tf.zeros_like(mask), tf.ones_like(mask))
				# n x k x m  [1,3]->[-inf,0,0,0,-inf,-inf...]
				mask = tf.to_float(mask) * (-1e13)
				# n x k x m [-inf,score,score,score,-inf,-inf...] for every span
				attention_score = mask + attention_score
				attention_weight = tf.math.softmax(attention_score, -1)
				# n x k x m and n x m x d -> n x k x d
				top_layer_args = tf.einsum('nkm,nmd->nkd', attention_weight, layer)
				# concat start and end token embedding
				top_layer_args = tf.concat([top_args_start, top_args_end, top_layer_args], axis=-1)
		elif args_embed_type == 'endpoint':
			# n x k x 2d
			top_layer_args = tf.concat([top_args_start, top_args_end], axis=-1)
		elif args_embed_type == 'max-poling':
			mask = tf.expand_dims(tf.expand_dims(tf.range(bucket_size), 0), 0)
			mask = tf.cast(tf.tile(mask, [batch_size, top_args_nums, 1]), tf.int64)
			mask_tmp1 = tf.where(tf.greater_equal(mask, top_args_idx_y), tf.ones_like(mask), tf.zeros_like(mask))
			mask_tmp2 = tf.where(tf.greater_equal(top_args_idx_z, mask), tf.ones_like(mask), tf.zeros_like(mask))
			mask = tf.where(tf.greater(mask_tmp1 + mask_tmp2, 1), tf.zeros_like(mask), tf.ones_like(mask))
			# n x k x m  [1,3]->[-inf,0,0,0,-inf,-inf...]
			mask = tf.to_float(mask) * (-1e13)
			# n x k x m [-inf,1,1,1,-inf,-inf...] for every span
			mask += tf.ones_like(mask)
			# n x k x m, n x m x d -> n x k x m x d
			top_layer_args = tf.einsum('nkm,nmd->nkmd', mask, layer)
			# n x k x m x d -> n x k x d
			top_layer_args = tf.reduce_max(top_layer_args, reduction_indices=-2, keep_dims=False)
		elif args_embed_type == 'coherent':
			top_args_start1, top_args_start2, top_args_start3, top_args_start4 = tf.split(top_args_start,tf.constant([560, 560, 40, 40]),-1)
			top_args_end1, top_args_end2, top_args_end3, top_args_end4 = tf.split(top_args_end,tf.constant([560, 560, 40, 40]),-1)
			top_layer_args = tf.concat([top_args_start1, top_args_end2,tf.expand_dims(tf.einsum('nkd,nkd->nk',top_args_start3,top_args_end4),-1)],-1)

		# n x p x m x m
		label_targets = self.placeholder
		# n x p x m x m
		unlabeled_targets = nn.greater(label_targets, 0)

		# n x m x m x p
		label_targets_trans = tf.transpose(label_targets, [0, 2, 3, 1])
		# n x k*p
		top_label_targets = tf.gather_nd(label_targets_trans, top_args_idx)
		# n x p x k
		top_label_targets = tf.transpose(tf.reshape(top_label_targets, [batch_size, top_args_nums, pred_nums]),
										 [0, 2, 1])

		# n x p x k
		top_unlabeled_targets = nn.greater(top_label_targets, 0)

		# n x m x m x p
		token_weights_trans = tf.transpose(token_weights, [0, 2, 3, 1])
		# n x k*p
		top_token_weights = tf.gather_nd(token_weights_trans, top_args_idx)
		# n x p x k
		top_token_weights = tf.transpose(tf.reshape(top_token_weights, [batch_size, top_args_nums, pred_nums]),
										 [0, 2, 1])

		with tf.variable_scope(variable_scope or self.field):

			with tf.variable_scope('Role'):
				layer_role = self.get_role_tensor(reuse=reuse)

			for i in six.moves.range(0, self.n_layers):
				with tf.variable_scope('ARGS-FC-%d' % i):
					top_layer_args = classifiers.hidden(top_layer_args, self.hidden_size,
														hidden_func=self.hidden_func,
														hidden_keep_prob=hidden_keep_prob)

				with tf.variable_scope('PRED-FC-%d' % i):
					layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
													 hidden_func=self.hidden_func,
													 hidden_keep_prob=hidden_keep_prob)
				if self.role_hidden:
					with tf.variable_scope('ROLE-FC-%d' % i):
						layer_role = classifiers.hidden(layer_role, self.hidden_size,
														hidden_func=self.hidden_func,
														hidden_keep_prob=hidden_keep_prob)

			with tf.variable_scope('Classifier'):

				if self.role_tensor:
					# n x p x k x role
					logits = classifiers.span_bilinear_classifier(
						layer_preds, top_layer_args, layer_role,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					# n x p x k x role
					logits = classifiers.diagonal_span_bilinear_classifier(
						layer_preds, top_layer_args, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

		# -----------------------------------------------------------
		# Compute the probabilities/cross entropy

		# (n x p x k x c) -> (n x p x k x c)
		label_probabilities = tf.nn.softmax(logits) * tf.to_float(tf.expand_dims(top_token_weights, axis=-1))
		# (n x k), (n x k x c), (n x k) -> ()
		label_loss = tf.losses.sparse_softmax_cross_entropy(top_label_targets, logits,
															weights=top_token_weights)
		# pdb.set_trace()
		# -----------------------------------------------------------
		# Compute the predictions/accuracy
		# (n x p x k x c) -> (n x p x k)
		# print('23333')
		predictions = tf.argmax(logits, axis=-1, output_type=tf.int32) * top_token_weights
		# (n x p x k) -> (n x p x k)
		unlabeled_predictions = nn.greater(predictions, 0)
		# (n x p x k) (*) (n x p x k) -> (n x p x k)
		unlabeled_true_positives = unlabeled_predictions * top_unlabeled_targets
		true_positives = nn.equal(top_label_targets, predictions) * top_unlabeled_targets
		# (n x p x k) -> ()
		n_predictions = tf.reduce_sum(unlabeled_predictions)
		# (n x p x m x m) -> ()
		n_targets = tf.reduce_sum(unlabeled_targets)
		n_unlabeled_true_positives = tf.reduce_sum(unlabeled_true_positives)
		n_true_positives = tf.reduce_sum(true_positives)
		# () - () -> ()
		n_unlabeled_false_positives = n_predictions - n_unlabeled_true_positives
		n_unlabeled_false_negatives = n_targets - n_unlabeled_true_positives
		n_false_positives = n_predictions - n_true_positives
		n_false_negatives = n_targets - n_true_positives
		# (n x p x m x m) -> (n)
		n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])
		n_unlabeled_true_positives_per_sequence = tf.reduce_sum(unlabeled_true_positives, axis=[1, 2])
		n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
		# (n) x 2 -> ()
		n_correct_unlabeled_sequences = tf.reduce_sum(
			nn.equal(n_unlabeled_true_positives_per_sequence, n_targets_per_sequence))
		n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

		outputs['unlabeled_targets'] = unlabeled_targets
		outputs['top_unlabeled_targets'] = top_unlabeled_targets
		outputs['label_targets'] = label_targets
		outputs['top_label_targets'] = top_label_targets

		outputs['probabilities'] = label_probabilities
		outputs['logits'] = logits

		outputs['loss'] = label_loss
		outputs['unlabeled_loss'] = label_loss

		outputs['unlabeled_predictions'] = unlabeled_predictions
		outputs['label_predictions'] = predictions

		outputs['n_unlabeled_true_positives'] = n_unlabeled_true_positives
		outputs['n_unlabeled_false_positives'] = n_unlabeled_false_positives
		outputs['n_unlabeled_false_negatives'] = n_unlabeled_false_negatives
		outputs['n_correct_unlabeled_sequences'] = n_correct_unlabeled_sequences
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences

		outputs['top_args_idx'] = top_args_idx

		return outputs

	# =============================================================
	def get_unfactored_trilinear_classifier(self, layer, preds, outputs, token_weights, variable_scope=None, reuse=False,
								 debug=False):
		""""""

		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear

		batch_size, bucket_size, input_size = nn.get_sizes(layer)
		one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
		layer_preds = tf.matmul(one_hot_pred, layer)
		pred_size = nn.get_sizes(layer_preds)[1]

		layer_args = layer
		layer_arge = layer

		with tf.variable_scope(variable_scope or self.field):

			with tf.variable_scope('Role'):
				layer_role = self.get_role_tensor(reuse=reuse)

			for i in six.moves.range(0, self.n_layers):
				with tf.variable_scope('ARGS-FC-%d' % i):
					layer_args = classifiers.hidden(layer_args, self.hidden_size,
													hidden_func=self.hidden_func,
													hidden_keep_prob=hidden_keep_prob)
				with tf.variable_scope('ARGE-FC-%d' % i):  # here is FNN? did not run
					layer_arge = classifiers.hidden(layer_arge, self.hidden_size,
													hidden_func=self.hidden_func,
													hidden_keep_prob=hidden_keep_prob)
				with tf.variable_scope('PRED-FC-%d' % i):
					layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
													 hidden_func=self.hidden_func,
													 hidden_keep_prob=hidden_keep_prob)
				if self.role_hidden:
					with tf.variable_scope('ROLE-FC-%d' % i):
						layer_role = classifiers.hidden(layer_role, self.hidden_size,
														hidden_func=self.hidden_func,
														hidden_keep_prob=hidden_keep_prob)

			with tf.variable_scope('Classifier'):
				if self.diagonal:
					logits = classifiers.span_diagonal_trilinear_classifier(
						layer_preds, layer_args, layer_arge, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					logits = classifiers.span_trilinear_classifier(
						layer_preds, layer_args, layer_arge, layer_role,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

		# -----------------------------------------------------------
		# Process the targets
		# (n x p x m x m)
		targets = self.placeholder

		# (n x p x m x m)
		unlabeled_targets = nn.greater(targets, 0)
		if self.top_k:
			select_place = outputs['logits_idx']
			targets_sel = tf.reshape(tf.gather_nd(targets, select_place),[batch_size, pred_size, -1])
			logits_sel = tf.reshape(tf.gather_nd(logits, select_place), [batch_size, pred_size, -1, len(self)])
			token_weights_sel = tf.reshape(tf.gather_nd(token_weights, select_place),[batch_size, pred_size, -1])


		# -----------------------------------------------------------
		# Compute the probabilities/cross entropy

		# (n x p x m x m x c) -> (n x p x m x m x c)
		probabilities = tf.nn.softmax(logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
		# (n x m x m), (n x m x m x c), (n x m x m) -> ()
		if self.top_k:
			label_loss = tf.losses.sparse_softmax_cross_entropy(targets_sel, logits_sel,
																weights=token_weights_sel)
		else:
			label_loss = tf.losses.sparse_softmax_cross_entropy(targets, logits,
																weights=token_weights)

		# -----------------------------------------------------------
		# Compute the predictions/accuracy
		# (n x p x m x m x c) -> (n x p x m x m)
		predictions = tf.argmax(logits, axis=-1, output_type=tf.int32) * token_weights
		# (n x p x m x m) -> (n x p x m x m)
		unlabeled_predictions = nn.greater(predictions, 0)

		# (n x p x m x m) (*) (n x p x m x m) -> (n x p x m x m)
		unlabeled_true_positives = unlabeled_predictions * unlabeled_targets
		true_positives = nn.equal(targets, predictions) * unlabeled_targets
		# (n x p x m x m) -> ()
		n_predictions = tf.reduce_sum(unlabeled_predictions)
		n_targets = tf.reduce_sum(unlabeled_targets)
		n_unlabeled_true_positives = tf.reduce_sum(unlabeled_true_positives)
		n_true_positives = tf.reduce_sum(true_positives)
		# () - () -> ()
		n_unlabeled_false_positives = n_predictions - n_unlabeled_true_positives
		n_unlabeled_false_negatives = n_targets - n_unlabeled_true_positives
		n_false_positives = n_predictions - n_true_positives
		n_false_negatives = n_targets - n_true_positives
		# (n x p x m x m) -> (n)
		n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2,3])
		n_unlabeled_true_positives_per_sequence = tf.reduce_sum(unlabeled_true_positives, axis=[1, 2,3])
		n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2,3])
		# (n) x 2 -> ()
		n_correct_unlabeled_sequences = tf.reduce_sum(
			nn.equal(n_unlabeled_true_positives_per_sequence, n_targets_per_sequence))
		n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

		# -----------------------------------------------------------
		# Populate the output dictionary

		rho = self.loss_interpolation
		outputs['unlabeled_targets'] = unlabeled_targets
		outputs['label_targets'] = self.placeholder
		outputs['label_probabilities'] = probabilities
		outputs['probabilities'] = probabilities

		outputs['loss'] = 2 * ((1 - rho) * outputs['unlabeled_loss'] + rho * label_loss)

		outputs['unlabeled_predictions'] = unlabeled_predictions
		outputs['label_predictions'] = predictions
		outputs['n_unlabeled_true_positives'] = n_unlabeled_true_positives
		outputs['n_unlabeled_false_positives'] = n_unlabeled_false_positives
		outputs['n_unlabeled_false_negatives'] = n_unlabeled_false_negatives
		outputs['n_correct_unlabeled_sequences'] = n_correct_unlabeled_sequences
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences

		outputs['unlabeled_targets'] = unlabeled_targets
		outputs['unlabeled_predictions'] = unlabeled_predictions
		return outputs

	@property
	def args_embed_type(self):
		try:
			return self._config.getstr(self, 'args_embed_type')
		except:
			return 'coherent'

	@property
	def span_diff(self):
		try:
			return self._config.getboolean(self, 'span_diff')
		except:
			return False

	@property
	def role_hidden(self):
		try:
			return self._config.getboolean(self, 'role_hidden')
		except:
			return False

	@property
	def role_tensor(self):
		try:
			return self._config.getboolean(self, 'role_tensor')
		except:
			return True

	@property
	def top_k(self):
		try:
			return self._config.getboolean(self, 'top_k')
		except:
			return False


#***************************************************************
class GraphTokenVocab(TokenVocab):
	""""""

	_depth = -1

	#=============================================================
	def __init__(self, *args, **kwargs):
		""""""
		kwargs['placeholder_shape'] = [None, None, None]
		super(GraphTokenVocab, self).__init__(*args, **kwargs)
		return

	#=============================================================
	def get_bilinear_discriminator(self, layer, token_weights, variable_scope=None, reuse=False):
		""""""

		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear
		with tf.variable_scope(variable_scope or self.classname):
			for i in six.moves.range(0, self.n_layers-1):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, 2*self.hidden_size,
																		 hidden_func=self.hidden_func,
																		 hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top' % i):
				layers = classifiers.hiddens(layer, 2*[self.hidden_size],
																	 hidden_func=self.hidden_func,
																	 hidden_keep_prob=hidden_keep_prob)
			layer1, layer2 = layers.pop(0), layers.pop(0)

			with tf.variable_scope('Discriminator'):
				if self.diagonal:
					logits = classifiers.diagonal_bilinear_discriminator(
						layer1, layer2,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					logits = classifiers.bilinear_discriminator(
						layer1, layer2,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

				#-----------------------------------------------------------
				# Process the targets
				# (n x m x m) -> (n x m x m)
				unlabeled_targets = nn.greater(self.placeholder, 0)

				#-----------------------------------------------------------
				# Compute probabilities/cross entropy
				# (n x m x m) -> (n x m x m)
				probabilities = tf.nn.sigmoid(logits)
				# (n x m x m), (n x m x m x c), (n x m x m) -> ()
				loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets, logits, weights=token_weights)

				#-----------------------------------------------------------
				# Compute predictions/accuracy
				# (n x m x m x c) -> (n x m x m)
				predictions = nn.greater(logits, 0, dtype=tf.int32) * token_weights
				# (n x m x m) (*) (n x m x m) -> (n x m x m)
				true_positives = predictions * unlabeled_targets
				# (n x m x m) -> ()
				n_predictions = tf.reduce_sum(predictions)
				n_targets = tf.reduce_sum(unlabeled_targets)
				n_true_positives = tf.reduce_sum(true_positives)
				# () - () -> ()
				n_false_positives = n_predictions - n_true_positives
				n_false_negatives = n_targets - n_true_positives
				# (n x m x m) -> (n)
				n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1,2])
				n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1,2])
				# (n) x 2 -> ()
				n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

		#-----------------------------------------------------------
		# Populate the output dictionary
		outputs = {}
		outputs['recur_layer'] = recur_layer
		outputs['unlabeled_targets'] = unlabeled_targets
		outputs['probabilities'] = probabilities
		outputs['unlabeled_loss'] = loss
		outputs['loss'] = loss

		outputs['unlabeled_predictions'] = predictions
		outputs['n_unlabeled_true_positives'] = n_true_positives
		outputs['n_unlabeled_false_positives'] = n_false_positives
		outputs['n_unlabeled_false_negatives'] = n_false_negatives
		outputs['n_correct_unlabeled_sequences'] = n_correct_sequences
		outputs['predictions'] = predictions
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences
		return outputs

	#=============================================================
	def get_bilinear_classifier(self, layer, outputs, token_weights, variable_scope=None, reuse=False, debug=False):
		""""""
		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear
		with tf.variable_scope(variable_scope or self.field):
			for i in six.moves.range(0, self.n_layers-1):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, 2*self.hidden_size,
																			hidden_func=self.hidden_func,
																			hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top'):
				layers = classifiers.hiddens(layer, 2*[self.hidden_size],
																		hidden_func=self.hidden_func,
																		hidden_keep_prob=hidden_keep_prob)
			layer1, layer2 = layers.pop(0), layers.pop(0)

			with tf.variable_scope('Classifier'):
				if self.diagonal:
					logits = classifiers.diagonal_bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					logits = classifiers.bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

		#-----------------------------------------------------------
		# Process the targets
		# (n x m x m)
		label_targets = self.placeholder
		unlabeled_predictions = outputs['unlabeled_predictions']
		unlabeled_targets = outputs['unlabeled_targets']

		#-----------------------------------------------------------
		# Process the logits
		# (n x m x c x m) -> (n x m x m x c)
		transposed_logits = tf.transpose(logits, [0,1,3,2])

		#-----------------------------------------------------------
		# Compute the probabilities/cross entropy
		# (n x m x m) -> (n x m x m x 1)
		head_probabilities = tf.expand_dims(tf.stop_gradient(outputs['head_probabilities']), axis=-1)
		# (n x m x m x c) -> (n x m x m x c)
		label_probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
		# (n x m x m), (n x m x m x c), (n x m x m) -> ()
		label_loss = tf.losses.sparse_softmax_cross_entropy(label_targets, transposed_logits, weights=token_weights*unlabeled_targets)

		#-----------------------------------------------------------
		# Compute the predictions/accuracy
		# (n x m x m x c) -> (n x m x m)
		#print('23333')
		predictions = tf.argmax(transposed_logits, axis=-1, output_type=tf.int32)
		# (n x m x m) (*) (n x m x m) -> (n x m x m)
		true_positives = nn.equal(label_targets, predictions) * unlabeled_predictions
		correct_label_tokens = nn.equal(label_targets, predictions) * unlabeled_targets
		# (n x m x m) -> ()
		n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
		n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
		n_true_positives = tf.reduce_sum(true_positives)
		n_correct_label_tokens = tf.reduce_sum(correct_label_tokens)
		# () - () -> ()
		n_false_positives = n_unlabeled_predictions - n_true_positives
		n_false_negatives = n_unlabeled_targets - n_true_positives
		# (n x m x m) -> (n)
		n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1,2])
		n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1,2])
		n_correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=[1,2])
		# (n) x 2 -> ()
		n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
		n_correct_label_sequences = tf.reduce_sum(nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))

		#-----------------------------------------------------------
		# Populate the output dictionary
		rho = self.loss_interpolation
		outputs['label_predictions'] = predictions
		outputs['label_probabilities'] = label_probabilities
		outputs['label_targets'] = label_targets
		outputs['probabilities'] = label_probabilities * head_probabilities
		outputs['label_loss'] = label_loss
		# outputs['label_logits'] = transposed_logits*tf.to_float(token_weights)
		# Combination of labeled loss and unlabeled loss
		outputs['loss'] = 2*((1-rho) * outputs['unlabeled_loss'] + rho * label_loss)
		# outputs['loss'] = label_loss * self.loss_rel_interpolation + outputs['loss'] * self.loss_edge_interpolation
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences
		outputs['n_correct_label_tokens'] = n_correct_label_tokens
		outputs['n_correct_label_sequences'] = n_correct_label_sequences

		return outputs

	@property
	def loss_rel_interpolation(self):
		try:
			return self._config.getfloat(self, 'loss_rel_interpolation')
		except:
			return self._config.getfloat(self, 'loss_interpolation')

	@property
	def loss_edge_interpolation(self):
		try:
			return self._config.getfloat(self, 'loss_edge_interpolation')
		except:
			return 1-self._config.getfloat(self, 'loss_interpolation')

	#=============================================================
	def get_unfactored_bilinear_classifier(self, layer, token_weights, variable_scope=None, reuse=False):
		""""""

		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear
		with tf.variable_scope(variable_scope or self.field):
			for i in six.moves.range(0, self.n_layers-1):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, 2*self.hidden_size,
																		 hidden_func=self.hidden_func,
																		 hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top'):
				layers = classifiers.hiddens(layer, 2*[self.hidden_size],
																		hidden_func=self.hidden_func,
																		hidden_keep_prob=hidden_keep_prob)
			layer1, layer2 = layers.pop(0), layers.pop(0)

			with tf.variable_scope('Classifier'):
				if self.diagonal:
					logits = classifiers.diagonal_bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					logits = classifiers.bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

				#-----------------------------------------------------------
				# Process the targets
				targets = self.placeholder
				# (n x m x m) -> (n x m x m)
				unlabeled_targets = nn.greater(targets, 0)

				#-----------------------------------------------------------
				# Process the logits
				# (n x m x c x m) -> (n x m x m x c)
				transposed_logits = tf.transpose(logits, [0,1,3,2])

				#-----------------------------------------------------------
				# Compute probabilities/cross entropy
				# (n x m x m x c) -> (n x m x m x c)
				probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
				# (n x m x m), (n x m x m x c), (n x m x m) -> ()
				loss = tf.losses.sparse_softmax_cross_entropy(targets, transposed_logits, weights=token_weights)

				#-----------------------------------------------------------
				# Compute predictions/accuracy
				# (n x m x m x c) -> (n x m x m)
				predictions = tf.argmax(transposed_logits, axis=-1, output_type=tf.int32) * token_weights
				# (n x m x m) -> (n x m x m)
				unlabeled_predictions = nn.greater(predictions, 0)

				if super(GraphTokenVocab, self).__getitem__('None')==5:
					unlabeled_targets = nn.greater(targets, 5)
					unlabeled_predictions = nn.greater(predictions, 5)


				# (n x m x m) (*) (n x m x m) -> (n x m x m)
				unlabeled_true_positives = unlabeled_predictions * unlabeled_targets
				true_positives = nn.equal(targets, predictions) * unlabeled_targets
				# (n x m x m) -> ()
				n_predictions = tf.reduce_sum(unlabeled_predictions)
				n_targets = tf.reduce_sum(unlabeled_targets)
				n_unlabeled_true_positives = tf.reduce_sum(unlabeled_true_positives)
				n_true_positives = tf.reduce_sum(true_positives)
				# () - () -> ()
				n_unlabeled_false_positives = n_predictions - n_unlabeled_true_positives
				n_unlabeled_false_negatives = n_targets - n_unlabeled_true_positives
				n_false_positives = n_predictions - n_true_positives
				n_false_negatives = n_targets - n_true_positives
				# (n x m x m) -> (n)
				n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1,2])
				n_unlabeled_true_positives_per_sequence = tf.reduce_sum(unlabeled_true_positives, axis=[1,2])
				n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1,2])
				# (n) x 2 -> ()
				n_correct_unlabeled_sequences = tf.reduce_sum(nn.equal(n_unlabeled_true_positives_per_sequence, n_targets_per_sequence))
				n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

		#-----------------------------------------------------------
		# Populate the output dictionary
		outputs = {}
		outputs['recur_layer'] = recur_layer
		outputs['unlabeled_targets'] = unlabeled_targets
		outputs['label_targets'] = self.placeholder
		outputs['label_probabilities'] = probabilities
		outputs['probabilities'] = probabilities
		outputs['unlabeled_loss'] = tf.constant(0.)
		outputs['loss'] = loss

		outputs['unlabeled_predictions'] = unlabeled_predictions
		outputs['label_predictions'] = predictions
		outputs['n_unlabeled_true_positives'] = n_unlabeled_true_positives
		outputs['n_unlabeled_false_positives'] = n_unlabeled_false_positives
		outputs['n_unlabeled_false_negatives'] = n_unlabeled_false_negatives
		outputs['n_correct_unlabeled_sequences'] = n_correct_unlabeled_sequences
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences

		outputs['unlabeled_targets'] = unlabeled_targets
		outputs['unlabeled_predictions'] = unlabeled_predictions
		return outputs

	def get_unfactored_bilinear_classifier_test(self, layer, outputs, refine, token_weights, variable_scope=None,
												reuse=False):
		""""""

		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear
		with tf.variable_scope(variable_scope or self.field):
			for i in six.moves.range(0, self.n_layers - 1):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, 2 * self.hidden_size,
											   hidden_func=self.hidden_func,
											   hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top'):
				layers = classifiers.hiddens(layer, 2 * [self.hidden_size],
											 hidden_func=self.hidden_func,
											 hidden_keep_prob=hidden_keep_prob)
			layer1, layer2 = layers.pop(0), layers.pop(0)

			with tf.variable_scope('Classifier'):
				if self.diagonal:
					logits = classifiers.diagonal_bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					logits = classifiers.bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

				# -----------------------------------------------------------
				# Process the targets
				targets = self.placeholder
				# (n x m x m) -> (n x m x m)
				# unlabeled_targets = nn.greater(targets, 0)

				# -----------------------------------------------------------
				# Process the logits
				# (n x m x c x m) -> (n x m x m x c)
				transposed_logits = tf.transpose(logits, [0, 1, 3, 2])

				# -----------------------------------------------------------
				# Compute probabilities/cross entropy
				# (n x m x m x c) -> (n x m x m x c)
				probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(
					tf.expand_dims(token_weights, axis=-1))

				unlabeled_predictions = outputs['unlabeled_predictions']
				unlabeled_targets = outputs['unlabeled_targets']

				# (n x m x m), (n x m x m x c), (n x m x m) -> ()
				if refine:
					label_loss = tf.losses.sparse_softmax_cross_entropy(targets, transposed_logits,
																  weights=token_weights*unlabeled_predictions)
				else:
					label_loss = tf.losses.sparse_softmax_cross_entropy(targets, transposed_logits, weights=token_weights)

				# -----------------------------------------------------------
				# Compute predictions/accuracy
				# (n x m x m x c) -> (n x m x m)
				predictions = tf.argmax(transposed_logits, axis=-1, output_type=tf.int32) * token_weights
				# (n x m x m) -> (n x m x m)
				unlabeled_predictions = nn.greater(predictions, 0)

				if super(GraphTokenVocab, self).__getitem__('None') == 5:
					unlabeled_targets = nn.greater(targets, 5)
					unlabeled_predictions = nn.greater(predictions, 5)

				# (n x m x m) (*) (n x m x m) -> (n x m x m)
				unlabeled_true_positives = unlabeled_predictions * unlabeled_targets
				true_positives = nn.equal(targets, predictions) * unlabeled_targets
				# (n x m x m) -> ()
				n_predictions = tf.reduce_sum(unlabeled_predictions)
				n_targets = tf.reduce_sum(unlabeled_targets)
				n_unlabeled_true_positives = tf.reduce_sum(unlabeled_true_positives)
				n_true_positives = tf.reduce_sum(true_positives)
				# () - () -> ()
				n_unlabeled_false_positives = n_predictions - n_unlabeled_true_positives
				n_unlabeled_false_negatives = n_targets - n_unlabeled_true_positives
				n_false_positives = n_predictions - n_true_positives
				n_false_negatives = n_targets - n_true_positives
				# (n x m x m) -> (n)
				n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2])
				n_unlabeled_true_positives_per_sequence = tf.reduce_sum(unlabeled_true_positives, axis=[1, 2])
				n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
				# (n) x 2 -> ()
				n_correct_unlabeled_sequences = tf.reduce_sum(
					nn.equal(n_unlabeled_true_positives_per_sequence, n_targets_per_sequence))
				n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

				correct_label_tokens = nn.equal(targets, predictions) * unlabeled_targets
				n_correct_label_tokens = tf.reduce_sum(correct_label_tokens)
				n_correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=[1, 2])
				n_correct_label_sequences = tf.reduce_sum(
					nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))

		# -----------------------------------------------------------
		# Populate the output dictionary
		rho = self.loss_interpolation
		balance_loss = outputs['balance_loss']
		outputs = {}
		outputs['recur_layer'] = recur_layer
		outputs['unlabeled_targets'] = unlabeled_targets
		outputs['label_targets'] = self.placeholder
		outputs['label_probabilities'] = probabilities
		outputs['probabilities'] = probabilities
		outputs['unlabeled_loss'] = tf.constant(0.)
		outputs['label_loss'] = label_loss
		# if refine:
		# 	outputs['loss'] = label_loss
		# else:
		# 	outputs['loss'] =  2*((1-rho) * outputs['unlabeled_loss'] + rho * label_loss)
		outputs['loss'] =  2*((1-rho) * label_loss + rho * balance_loss)

		outputs['unlabeled_predictions'] = unlabeled_predictions
		outputs['label_predictions'] = predictions
		outputs['n_unlabeled_true_positives'] = n_unlabeled_true_positives
		outputs['n_unlabeled_false_positives'] = n_unlabeled_false_positives
		outputs['n_unlabeled_false_negatives'] = n_unlabeled_false_negatives
		outputs['n_correct_unlabeled_sequences'] = n_correct_unlabeled_sequences
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_label_tokens'] = n_correct_label_tokens
		outputs['n_correct_label_sequences'] = n_correct_label_sequences
		outputs['n_correct_sequences'] = n_correct_sequences

		outputs['unlabeled_targets'] = unlabeled_targets
		outputs['unlabeled_predictions'] = unlabeled_predictions
		return outputs

	# =============================================================
	def get_unfactored_bilinear_classifier_balance(self, layer, token_weights, variable_scope=None, reuse=False):
		""""""

		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear
		with tf.variable_scope(variable_scope or self.field):
			for i in six.moves.range(0, self.n_layers - 1):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, 2 * self.hidden_size,
											   hidden_func=self.hidden_func,
											   hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top'):
				layers = classifiers.hiddens(layer, 2 * [self.hidden_size],
											 hidden_func=self.hidden_func,
											 hidden_keep_prob=hidden_keep_prob)
			layer1, layer2 = layers.pop(0), layers.pop(0)

			with tf.variable_scope('Classifier'):
				if self.diagonal:
					logits = classifiers.diagonal_bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					logits = classifiers.bilinear_classifier(
						layer1, layer2, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

				# -----------------------------------------------------------
				# Process the targets
				targets = self.placeholder
				# (n x m x m) -> (n x m x m)
				unlabeled_targets = nn.greater(targets, 0)

				# -----------------------------------------------------------
				# Process the logits
				# (n x m x c x m) -> (n x m x m x c)
				transposed_logits = tf.transpose(logits, [0, 1, 3, 2])

				# -----------------------------------------------------------
				# Compute probabilities/cross entropy
				# (n x m x m x c) -> (n x m x m x c)
				probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(
					tf.expand_dims(token_weights, axis=-1))
				# (n x m x m), (n x m x m x c), (n x m x m) -> ()
				loss = tf.losses.sparse_softmax_cross_entropy(unlabeled_targets, transposed_logits, weights=token_weights)

				# -----------------------------------------------------------
				# Compute predictions/accuracy
				# (n x m x m x c) -> (n x m x m)
				predictions = tf.argmax(transposed_logits, axis=-1, output_type=tf.int32) * token_weights
				# (n x m x m) -> (n x m x m)
				unlabeled_predictions = nn.greater(predictions, 0)

				if super(GraphTokenVocab, self).__getitem__('None') == 5:
					unlabeled_targets = nn.greater(targets, 5)
					unlabeled_predictions = nn.greater(predictions, 5)

				# (n x m x m) (*) (n x m x m) -> (n x m x m)
				unlabeled_true_positives = unlabeled_predictions * unlabeled_targets
				true_positives = nn.equal(targets, predictions) * unlabeled_targets
				# (n x m x m) -> ()
				n_predictions = tf.reduce_sum(unlabeled_predictions)
				n_targets = tf.reduce_sum(unlabeled_targets)
				n_unlabeled_true_positives = tf.reduce_sum(unlabeled_true_positives)
				n_true_positives = tf.reduce_sum(true_positives)
				# () - () -> ()
				n_unlabeled_false_positives = n_predictions - n_unlabeled_true_positives
				n_unlabeled_false_negatives = n_targets - n_unlabeled_true_positives
				n_false_positives = n_predictions - n_true_positives
				n_false_negatives = n_targets - n_true_positives
				# (n x m x m) -> (n)
				n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2])
				n_unlabeled_true_positives_per_sequence = tf.reduce_sum(unlabeled_true_positives, axis=[1, 2])
				n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
				# (n) x 2 -> ()
				n_correct_unlabeled_sequences = tf.reduce_sum(
					nn.equal(n_unlabeled_true_positives_per_sequence, n_targets_per_sequence))
				n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

		# -----------------------------------------------------------
		# Populate the output dictionary
		outputs = {}
		outputs['recur_layer'] = recur_layer
		outputs['unlabeled_targets'] = unlabeled_targets
		outputs['label_targets'] = self.placeholder
		outputs['label_probabilities'] = probabilities
		outputs['probabilities'] = probabilities
		outputs['unlabeled_loss'] = tf.constant(0.)
		outputs['balance_loss'] = loss

		outputs['unlabeled_predictions'] = unlabeled_predictions
		outputs['label_predictions'] = predictions
		outputs['n_unlabeled_true_positives'] = n_unlabeled_true_positives
		outputs['n_unlabeled_false_positives'] = n_unlabeled_false_positives
		outputs['n_unlabeled_false_negatives'] = n_unlabeled_false_negatives
		outputs['n_correct_unlabeled_sequences'] = n_correct_unlabeled_sequences
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences

		outputs['unlabeled_targets'] = unlabeled_targets
		outputs['unlabeled_predictions'] = unlabeled_predictions
		return outputs

	#=============================================================
	def _count(self, node):
		if node not in ('_', ''):
			node = node.split('|')
			for edge in node:
				edge = edge.split(':', 1)
				head, rel = edge
				self.counts[rel] += 1
		return

	#=============================================================
	def add(self, token):
		""""""

		return self.index(token)

	#=============================================================
	# token should be: 1:rel|2:acl|5:dep
	def index(self, token):
		""""""

		nodes = []
		if token != '_':
			token = token.split('|')
			for edge in token:
				head, semrel = edge.split(':', 1)
				nodes.append( (int(head), super(GraphTokenVocab, self).__getitem__(semrel)) )
		return nodes

	#=============================================================
	# index should be [(1, 12), (2, 4), (5, 2)]
	def token(self, index):
		""""""

		nodes = []
		for (head, semrel) in index:
			nodes.append('{}:{}'.format(head, super(GraphTokenVocab, self).__getitem__(semrel)))
		return '|'.join(nodes)

	#=============================================================
	def get_root(self):
		""""""

		return '_'


	#=============================================================
	def __getitem__(self, key):
		if isinstance(key, six.string_types):
			nodes = []
			if key != '_':
				token = key.split('|')
				for edge in token:
					head, rel = edge.split(':', 1)
					nodes.append( (int(head), super(GraphTokenVocab, self).__getitem__(rel)) )
			return nodes
		elif hasattr(key, '__iter__'):
			if len(key) > 0 and hasattr(key[0], '__iter__'):
				if len(key[0]) > 0 and isinstance(key[0][0], six.integer_types + (np.int32, np.int64)):
					nodes = []
					for (head, rel) in key:
						nodes.append('{}:{}'.format(head, super(GraphTokenVocab, self).__getitem__(rel)))
					return '|'.join(nodes)
				else:
					return [self[k] for k in key]
			else:
				return '_'
		else:
			raise ValueError('key to GraphTokenVocab.__getitem__ must be (iterable of) strings or iterable of integers')

#***************************************************************
class SecondGraphTokenVocab(TokenVocab):
	_depth = -1

	# =============================================================
	def __init__(self, *args, **kwargs):
		""""""
		kwargs['placeholder_shape'] = [None, None, None]
		super(SecondGraphTokenVocab, self).__init__(*args, **kwargs)
		return


	# =============================================================
	def get_trilinear_classifier_old(self, layer, outputs, unary_predictions, token_weights, variable_scope=None, reuse=False, debug=False):
		""""""
		# pdb.set_trace()
		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear
		hidden_sizes = 2 * self.unary_hidden_size + 2 * self.sib_hidden_size  # sib_head, sib_dep
		with tf.variable_scope(variable_scope or self.field):
			for i in six.moves.range(0, self.n_layers - 1):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, hidden_sizes,
											   hidden_func=self.hidden_func,
											   hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top'):
				layers = classifiers.hiddens(layer, 2 * [self.unary_hidden_size] + 2 * [self.sib_hidden_size],
											 hidden_func=self.hidden_func,
											 hidden_keep_prob=hidden_keep_prob)
			unary_head, unary_dep, sib_head, sib_dep = layers.pop(0), layers.pop(0), layers.pop(0), layers.pop(0)

			with tf.variable_scope('Classifier'):
				if self.diagonal:
					# (n x m x o x m)
					unary_logits = classifiers.diagonal_bilinear_classifier(
						unary_head, unary_dep, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					# (n x m x o x m)
					unary_logits = classifiers.bilinear_classifier(
						unary_head, unary_dep, len(self),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

				# (n x m x o x m) -> (n x m x m x o)
				unary_logits = tf.transpose(unary_logits, perm=[0, 1, 3, 2])
				unary_logits_shape = nn.get_sizes(unary_logits)  # n, m, m, o

				# c_roles = ['A0', 'A1' ,'A2', 'A3', 'AM-TMP', 'AM-MNR', 'AM-ADV', 'AM-LOC', 'AM-DIS', 'AM-EXT']
				c_roles = ['A0', 'A1']

				if self.use_sib:
					idx = [super(SecondGraphTokenVocab, self).__getitem__(role) for role in c_roles]
					one_hot_c_roles = tf.one_hot(idx, len(self))
					unary_logits_l_mask = 1-tf.reduce_sum(one_hot_c_roles,axis=0) # 1 x o
					# n x m x m x o
					unary_logits_l_roles = unary_logits * tf.expand_dims(tf.expand_dims(tf.expand_dims(unary_logits_l_mask,axis=0),axis=0),axis=0)

					# n x m x m x 10
					unary_logits_c_roles = tf.matmul(tf.reshape(unary_logits,[-1,len(self)]),tf.transpose(one_hot_c_roles))
					unary_logits_c_roles = tf.reshape(unary_logits_c_roles, [unary_logits_shape[0], unary_logits_shape[1], unary_logits_shape[2], len(c_roles)])

					with tf.variable_scope('Sibling'):
						# layer = (n x 10 x 10 x m x m x m)
						sib_logits = classifiers.trilinear_classifier(
							sib_head, sib_dep, sib_dep, len(c_roles),
							hidden_keep_prob=hidden_keep_prob,
							add_linear=add_linear)
						# (n x m x m x m x 10 x 10)
						sib_logits = tf.transpose(sib_logits, perm=[0,3,4,5,1,2])
				# if self.use_gp:
				#   with tf.variable_scope('GrandParents'):
				#     layer_gp = classifiers.trilinear_classifier(
				#       layer1, layer2, layer3, len(self),
				#       hidden_keep_prob=hidden_keep_prob,
				#       add_linear=add_linear)
				# if self.use_cop:
				#   with tf.variable_scope('CoParents'):
				#     layer_cop = classifiers.trilinear_classifier(
				#       layer1, layer2, layer3, len(self),
				#       hidden_keep_prob=hidden_keep_prob,
				#       add_linear=add_linear)


			# mask words whose role is not our choice roles
			# n x mp x ma
			# unary_predictions = tf.argmax(unary_logits, axis=-1, output_type=tf.int32)
			# # n x ma x mp -> n x mp x ma
			# unary_predictions = tf.transpose(unary_predictions, perm=[0,2,1])

			# n x mc x mp
			mask3D = tf.zeros_like(unary_predictions)
			for i in idx:
				mask3D += tf.where(tf.equal(unary_predictions,i),tf.ones_like(unary_predictions),tf.zeros_like(unary_predictions))
			mask4D = tf.expand_dims(mask3D,axis=-1)
			token_weights4D = tf.expand_dims(token_weights,axis=-1)
			# n x mc x mp x c_roles with masking word without c_roles
			unary_logits_c_roles_mask = unary_logits_c_roles * tf.to_float(mask4D) * tf.to_float(token_weights4D)


			# unary_potential=-unary
			# n x mp x ma x 10
			q_value = unary_logits_c_roles_mask
			# o is label of Q(a,b), i is label of Q(a,c)
			# 1 sibling (n x o x i x ma x mb x mc) * (n x i x ma x mc) -> (n x o x i x ma x mb)
			# 2 grand parent (n x o x i x ma x mb x mc) * (n x i x mb x mc) -> (n x o x i x ma x mb)
			# 3 coparent (n x o x i x ma x mb x mc) * (n x i x mc x mb) -> (n x o x i x ma x mb)

			# binary_shape = nn.get_sizes(layer_sib)
			if debug:
				outputs['q_value_orig'] = q_value
			for i in range(int(self.num_iteration)):
				# n x mc x mp x c_roles
				q_value = tf.nn.softmax(q_value, -1)
				q_value = q_value * tf.to_float(mask4D) * tf.to_float(token_weights4D)
				if debug and i == 0:
					outputs['q_value_old'] = q_value
				# q_value (n x mc x mp x 10) * sib_logits (n x m x m x m x 10 x 10) -> F_temp (n x m x m x m x 10)
				if self.use_sib:
					# second_temp_sib = tf.einsum('niac,noiabc->noiab', q_value, layer_sib)
					# n x mk x mp x ma x 10
					F_temp_sib = tf.einsum('nkpior,nkpr->nkpio', sib_logits, q_value)
					# n x mp x 10 x mk x ma
					F_temp_sib = tf.transpose(F_temp_sib, perm=[0, 2, 4, 1, 3])
					F_temp_upper = (F_temp_sib - tf.linalg.band_part(F_temp_sib, 0, -1)) + tf.transpose(
						(F_temp_sib - tf.linalg.band_part(F_temp_sib, -1, 0)),
						perm=[0, 1, 2, 4, 3])
					F_temp_lower = (F_temp_sib - tf.linalg.band_part(F_temp_sib, -1, 0)) + tf.transpose(
						(F_temp_sib - tf.linalg.band_part(F_temp_sib, 0, -1)),
						perm=[0, 1, 2, 4, 3])
					F_temp_upper = tf.transpose(F_temp_upper, perm=[0, 1, 3, 4, 2])
					F_temp_lower = tf.transpose(F_temp_lower, perm=[0, 1, 3, 4, 2])
					# n x mp x ma x mb x 10 : a->i; b->k
					F_temp_sib = F_temp_upper + F_temp_lower
					# ->root 0 ; ->pad-> 0
					# F_temp_sib = F_temp_sib * token_weights4D
					# n x o x mp x ma x mb
					# F_temp_sib = tf.transpose(F_temp_sib, perm=[0, 4, 1, 2, 3])

				else:
					F_temp_sib = 0
				# if self.use_gp:
				#   #'''
				#   #a->b->c
				#   second_temp_gp = tf.einsum('nibc,noiabc->noiab', q_value, layer_gp)
				#   #c->a->b
				#   second_temp_gp2 = tf.einsum('nica,noiabc->noiab', q_value, layer_gp)
				# else:
				#   second_temp_gp=0
				#   second_temp_gp2=0
				# if self.use_cop:
				#   with tf.device('/device:GPU:2'):
				#     second_temp_cop = tf.einsum('nicb,noiabc->noiab', q_value, layer_cop)
				# else:
				#   second_temp_cop=0
				'''
				if self.self_minus:
	  
				  if self.use_sib:
					#-----------------------------------------------------------
					#sib part algorithm checked
					# minus all a = b = c part
					#(n x i x ma x mb) -> (n x i x ma) -> (n x i x ma x 1) -> (n x 1 x i x ma x 1) | (n x o x i x ma x mb x mc) -> (n x o x i x mb x ma x mc) -> (n x o x i x mb x ma) -> (n x o x i x ma x mb)
					#Q(a,a)*p(a,b,a) 
					#TODO: is this right?
					diag_sib1 = tf.expand_dims(tf.expand_dims(tf.linalg.diag_part(q_value),-1),1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_sib,perm=[0,1,2,4,3,5])),perm=[0,1,2,4,3])
					# (n x o x i x ma x mb x mc) -> (n x o x i x ma x mb)
					#Q(a,b)*p(a,b,b)
					diag_sib2 = tf.expand_dims(q_value,1) * tf.linalg.diag_part(layer_sib)
	  
					second_temp_sib = second_temp_sib - diag_sib1 - diag_sib2
				  if self.use_gp:
					#(n x o x i x ma x mb x mc) -> (n x o x i x mb x ma x mc) -> (n x o x i x mb x ma) -> (n x o x i x ma x mb)
					#Q(b,a)*p(a,b,a)
					diag_gp1 = tf.expand_dims(tf.transpose(q_value,perm=[0,1,3,2]),1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp,perm=[0,1,2,4,3,5])),perm=[0,1,2,4,3])
					#(n x o x i x ma x mb) -> (n x o x i x mb) -> (n x o x i x 1 x mb) | (n x o x i x ma x mb x mc) -> (n x o x i x ma x mb)
					#Q(b,b)*p(a,b,b)
					diag_gp2 = tf.expand_dims(tf.expand_dims(tf.linalg.diag_part(q_value),-2),1) * tf.linalg.diag_part(layer_gp)
	  
					#(n x o x i x ma x mb) -> (n x o x i x ma) -> (n x o x i x ma x 1) | (n x o x i x ma x mb x mc) -> (n x o x i x mb x ma x mc) -> (n x o x i x mb x ma) -> (n x o x i x ma x mb)
					#????? Is this right? Yes
					#Q(a,a)*p(a,b,a)
					diag_gp21 = tf.expand_dims(tf.expand_dims(tf.linalg.diag_part(q_value),-1),1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_gp,perm=[0,1,2,4,3,5])),perm=[0,1,2,4,3])
					#(n x o x i x ma x mb) -> (n x o x i x mb x ma) | (n x o x i x ma x mb x mc) -> (n x o x i x ma x mb)
					#Q(b,a)*p(a,b,b)
					diag_gp22 = tf.expand_dims(tf.transpose(q_value,perm=[0,1,3,2]),1) * tf.linalg.diag_part(layer_gp)
	  
					second_temp_gp = second_temp_gp - diag_gp1 - diag_gp2
					#c->a->b
					second_temp_gp2 = second_temp_gp2 - diag_gp21 - diag_gp22
					#second_temp_gp=second_temp_gp+second_temp_gp2
				  if self.use_cop:
					with tf.device('/device:GPU:2'):
					  #(n x o x i x ma x mb x mc) -> (n x o x i x mb x ma x mc) -> (n x o x i x mb x ma) -> (n x o x i x ma x mb)
					  #Q(a,b)*p(a,b,a)
					  diag_cop1 = tf.expand_dims(q_value,1) * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_cop,perm=[0,1,2,4,3,5])),perm=[0,1,2,4,3])
					  #(n x o x i x ma x mb) -> (n x o x i x mb) -> (n x o x i x 1 x mb) | (n x o x i x ma x mb x mc) -> (n x o x i x ma x mb)
					  #Q(b,b)*p(a,b,b)
					  diag_cop2 = tf.expand_dims(tf.expand_dims(tf.linalg.diag_part(q_value),-2),1) * tf.linalg.diag_part(layer_cop)
	  
					second_temp_cop = second_temp_cop - diag_cop1 - diag_cop2
	  
				pdb.set_trace()
				'''
				# (n x o x mp x mh1 x mh2)
				# second_temp = second_temp_sib + second_temp_gp + second_temp_gp2 + second_temp_cop
				# n x mp x mk x ma x 10
				F = F_temp_sib

				# Second order potential update function
				# second_temp=unary_potential - second_temp
				# (n x mp x ma x 10)
				q_value = unary_logits_c_roles_mask + tf.transpose(tf.reduce_sum(F, -3),perm=[0,2,1,3])
				# q_value=tf.stack([unary_potential[:,:,0,:],second_temp],axis=2)
				if debug:
					outputs['q_value' + str(i)] = q_value


			# n x mc x mp x 10: ma masked
			q_value = q_value * tf.to_float(mask4D) * tf.to_float(token_weights4D)
			unary_logits_c_roles_init = unary_logits_c_roles * tf.to_float((1-mask4D)) * tf.to_float(token_weights4D)
			# n x mc x mp x 10
			q_value = q_value + unary_logits_c_roles_init

			# n x mc x mp x 10 -> n x mc x mp x o
			q_value = tf.reshape(tf.matmul(tf.reshape(q_value,[-1,len(c_roles)]),one_hot_c_roles),[unary_logits_shape[0], unary_logits_shape[1], unary_logits_shape[2],-1])

		transposed_logits = unary_logits_l_roles + q_value

		# -----------------------------------------------------------
		# Process the targets
		# (n x m x m)
		label_targets = self.placeholder
		unlabeled_predictions = outputs['unlabeled_predictions']
		unlabeled_targets = outputs['unlabeled_targets']

		# -----------------------------------------------------------
		# Process the logits
		# if self.transposed:
		# 	# need transpose as the target is the transpose
		# 	# (n x o x ma x mb) -> (n x mb x ma x o)
		# 	transposed_logits = tf.transpose(q_value, [0, 3, 2, 1])
		# else:
		# 	# (n x o x ma x mb) -> (n x ma x mb x o)
		# 	transposed_logits = tf.transpose(q_value, [0, 2, 3, 1])

		# -----------------------------------------------------------
		# Compute the probabilities/cross entropy
		# (n x m x m) -> (n x m x m x 1)
		head_probabilities = tf.expand_dims(tf.stop_gradient(outputs['head_probabilities']), axis=-1)
		# (n x m x m x c) -> (n x m x m x c)
		label_probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
		# (n x m x m), (n x m x m x c), (n x m x m) -> ()
		label_loss = tf.losses.sparse_softmax_cross_entropy(label_targets, transposed_logits,
															weights=token_weights * unlabeled_targets)

		# -----------------------------------------------------------
		# Compute the predictions/accuracy
		# (n x m x m x c) -> (n x m x m)
		predictions = tf.argmax(transposed_logits, axis=-1, output_type=tf.int32)
		# (n x m x m) (*) (n x m x m) -> (n x m x m)
		true_positives = nn.equal(label_targets, predictions) * unlabeled_predictions
		correct_label_tokens = nn.equal(label_targets, predictions) * unlabeled_targets
		# (n x m x m) -> ()
		n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
		n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
		n_true_positives = tf.reduce_sum(true_positives)
		n_correct_label_tokens = tf.reduce_sum(correct_label_tokens)
		# () - () -> ()
		n_false_positives = n_unlabeled_predictions - n_true_positives
		n_false_negatives = n_unlabeled_targets - n_true_positives
		# (n x m x m) -> (n)
		n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2])
		n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
		n_correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=[1, 2])
		# (n) x 2 -> ()
		n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
		n_correct_label_sequences = tf.reduce_sum(nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))

		# -----------------------------------------------------------
		# Populate the output dictionary
		rho = self.loss_interpolation

		#=================debug======================================
		outputs['unary_logits'] = unary_logits
		outputs['unary_logits_l_roles'] = unary_logits_l_roles
		outputs['unary_logits_c_roles'] = unary_logits_c_roles
		outputs['unary_logits_c_roles_init'] = unary_logits_c_roles_init
		outputs['unary_logits_c_roles_mask'] = unary_logits_c_roles_mask
		outputs['q_value'] = q_value
		outputs['token_weights'] = token_weights
		#=================debug======================================

		outputs['label_targets'] = label_targets
		outputs['unary_predictions'] = unary_predictions
		outputs['predictions'] = predictions
		outputs['probabilities'] = label_probabilities * head_probabilities
		outputs['label_probabilities'] = label_probabilities
		outputs['label_prob'] = label_probabilities
		outputs['label_loss'] = label_loss
		# Combination of labeled loss and unlabeled loss
		outputs['loss'] = 2 * ((1 - rho) * outputs['loss'] + rho * label_loss)

		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences
		outputs['n_correct_label_tokens'] = n_correct_label_tokens
		outputs['n_correct_label_sequences'] = n_correct_label_sequences
		return outputs

	# =============================================================
	def get_trilinear_classifier_new(self, layer, outputs, tokens, variable_scope=None,
								 reuse=False, debug=False):
		""""""
		# pdb.set_trace()
		# c_roles = ['A0', 'A1' ,'A2', 'A3', 'AM-TMP', 'AM-MNR', 'AM-ADV', 'AM-LOC', 'AM-DIS', 'AM-EXT']
		c_roles = ['None','A1']

		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear
		hidden_sizes = 2 * self.unary_hidden_size + 2 * self.sib_hidden_size  # sib_head, sib_dep
		with tf.variable_scope(variable_scope or self.field):
			for i in six.moves.range(0, self.n_layers - 1):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, hidden_sizes,
											   hidden_func=self.hidden_func,
											   hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top'):
				layers = classifiers.hiddens(layer, 2 * [self.unary_hidden_size] + 2 * [self.sib_hidden_size],
											 hidden_func=self.hidden_func,
											 hidden_keep_prob=hidden_keep_prob)
			unary_head, unary_dep, sib_head, sib_dep = layers.pop(0), layers.pop(0), layers.pop(0), layers.pop(0)

			with tf.variable_scope('Classifier'):
				if self.diagonal:
					# (n x mp x o x mc)
					unary_logits = classifiers.diagonal_bilinear_classifier(
						unary_head, unary_dep, len(c_roles),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)
				else:
					# (n x mp x o x mc)
					unary_logits = classifiers.bilinear_classifier(
						unary_head, unary_dep, len(c_roles),
						hidden_keep_prob=hidden_keep_prob,
						add_linear=add_linear)

				# (n x mp x o x mc) -> (n x mp x mc x o)
				unary_logits = tf.transpose(unary_logits, perm=[0, 1, 3, 2])
				unary_logits_shape = nn.get_sizes(unary_logits)  # n, mp, mc, 10

				idx = [super(SecondGraphTokenVocab, self).__getitem__(role) for role in c_roles]
				one_hot_c_roles = tf.one_hot(idx, len(self))

				if self.use_sib:

					with tf.variable_scope('Sibling'):
						# n x mp x mc x mc x 10 x 10
						sib_logits = classifiers.diagonal_trilinear_classifier1(
							sib_head, sib_dep, sib_dep, len(c_roles),
							hidden_keep_prob=hidden_keep_prob)


			# n x mc x mp
			# unary_predictions: (n x mc x mp) -> (n x mp x mc)
			token_weights = tokens['token_weights']
			token_weights3D = tokens['token_weights3D']
			if self.factorized:
				unary_predictions = tf.stop_gradient(outputs['label_predictions']) * \
									tf.stop_gradient(outputs['unlabeled_predictions'])
			else:
				unary_predictions = tf.stop_gradient(outputs['label_predictions'])
			unary_predictions = tf.transpose(unary_predictions,perm=[0,2,1])
			# n x mp x mc
			mask3D = tf.zeros_like(unary_predictions)
			for i in idx:
				mask3D += tf.where(tf.equal(unary_predictions, i), tf.ones_like(unary_predictions),
								   tf.zeros_like(unary_predictions))
			mask4D = tf.expand_dims(mask3D, axis=-1)
			# token_weights3D is (n x mc x mp), token_weights4D is (n x mp x mc x 1)
			token_weights4D = tf.expand_dims(tf.transpose(token_weights3D,perm=[0,2,1]), axis=-1)
			# n x mp x mc x 10 mask fixed words whose predictions are not in our selected roles
			unary_logits_unfixed_words = unary_logits * tf.to_float(mask4D) * tf.to_float(token_weights4D)
			# set sib_logits symmetrical (means c1c2r1r2=c2c1r2r1 but c1c2r1r2!=c1c2r2r1) and mask root and padding
			if self.use_sib:
				token_weights_sib = tf.cast(tf.expand_dims(tf.transpose(token_weights3D, [0, 2, 1]), axis=-1) * tf.expand_dims(
						tf.expand_dims(token_weights, axis=1), axis=1), dtype=tf.float32)
				sib_logits = sib_logits * tf.expand_dims(tf.expand_dims(token_weights_sib,-1),-1)
				sib_logits_trans = tf.transpose(sib_logits, perm=[0, 1, 4, 5, 2, 3])
				# make m1m2r1r2 = m2m1r2r1
				sib_logits_trans = sib_logits_trans - tf.linalg.band_part(sib_logits_trans, -1, 0) + tf.transpose(
					tf.linalg.band_part(sib_logits_trans, 0, -1), perm=[0, 1, 3, 2, 5, 4])
				# n x mp x mc1 x mc2 x r x r
				sib_logits_sym = tf.transpose(sib_logits_trans, perm=[0, 1, 4, 5, 2, 3])

			# n x mp x mc x 10
			q_value = unary_logits_unfixed_words

			for i in range(int(self.num_iteration)):
				# n x mp x mc x 10
				q_value = tf.nn.softmax(q_value, -1)
				q_value = q_value * tf.to_float(mask4D) * tf.to_float(token_weights4D)

				if self.use_sib:
					# q_value (n x mp x mc x 10) * sib_logits (n x mp x mc1 x mc2 x 10 x 10) -> F_temp_sib (n x mp x mc1 x mc2 x 10)
					F_temp_sib = tf.einsum('npikor,npkr->npiko', sib_logits_sym, q_value)
					# 对esuim运算后得到的 n x mp x mc1 x mc2 x 10 矩阵的对角线设为0
					diag_ones = tf.matrix_diag(tf.ones([unary_logits_shape[0], unary_logits_shape[1], unary_logits_shape[2]]))
					diag_mask = 1 - diag_ones
					F_temp_sib = F_temp_sib * tf.expand_dims(diag_mask, -1)
				else:
					F_temp_sib = 0

				# n x mp x mc1 x mc2 x 10
				F = F_temp_sib

				# (n x mp x mc1 x 10)
				q_value = unary_logits_unfixed_words + tf.reduce_sum(F, -2)


			# n x mp x mc1 x 10  masked
			# q_value = q_value * tf.to_float(mask4D) * tf.to_float(token_weights4D)

			# (n x mp x mc x 10) -> (n x mp x mc x o)
			logits = tf.reshape(tf.matmul(tf.reshape(q_value, [-1, len(c_roles)]), one_hot_c_roles),
								 [unary_logits_shape[0], unary_logits_shape[1], unary_logits_shape[2], -1])

		# (n x mp x mc x o) -> (n x mc x mp x o)
		transposed_logits = tf.transpose(logits, perm=[0,2,1,3])

		# -----------------------------------------------------------
		# Process the targets
		# (n x mc x mp)
		label_targets = self.placeholder
		unlabeled_predictions = tf.stop_gradient(outputs['unlabeled_predictions'])
		unlabeled_targets = tf.stop_gradient(outputs['unlabeled_targets'])

		# init token_weights3D is (n x mc x mp), token_weights4D is (n x mp x mc x1), mask3D is (n x mp x mc), mask4D is (n x mp x mc x 1)
		# due targets is (n x mc x mp), transpose token_weights4D, mask3D and mask4D
		token_weights4D = tf.transpose(token_weights4D, perm=[0,2,1,3])
		mask3D = tf.transpose(mask3D, perm=[0,2,1])
		mask4D = tf.transpose(mask4D, perm=[0,2,1,3])


		# -----------------------------------------------------------
		# Compute the probabilities/cross entropy
		# (n x mc x mp) -> (n x mc x mp x 1)
		if self.factorized:
			head_probabilities = tf.expand_dims(tf.stop_gradient(outputs['head_probabilities']), axis=-1)
		# (n x mc x mp x c) -> (n x mc x mp x c)
		fixed_label_probabilities = tf.stop_gradient(outputs['label_probabilities']) * tf.to_float((1-mask4D)) * tf.to_float(token_weights4D)
		unfixed_label_probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights3D, axis=-1)) * tf.to_float(mask4D)
		label_probabilities = fixed_label_probabilities + unfixed_label_probabilities
		# (n x m x m), (n x m x m x c), (n x m x m) -> () just compute label_loss of unfixed words
		if self.factorized:
			label_loss = tf.losses.sparse_softmax_cross_entropy(label_targets, transposed_logits,
															weights=token_weights3D * unlabeled_targets * mask3D)
		else:
			label_loss = tf.losses.sparse_softmax_cross_entropy(label_targets, transposed_logits,
																weights=token_weights3D * mask3D)
		# -----------------------------------------------------------
		# Compute the predictions/accuracy
		# predictions of fixed word from first order
		fixed_label_predictions = tf.stop_gradient(outputs['label_predictions']) * (1-mask3D)
		# (n x m x m x c) -> (n x m x m)
		unfixed_label_predictions = tf.argmax(transposed_logits, axis=-1, output_type=tf.int32) * mask3D
		label_predictions = fixed_label_predictions + unfixed_label_predictions
		if not self.factorized:
			unlabeled_predictions = nn.greater(label_predictions, 5)
		# (n x m x m) (*) (n x m x m) -> (n x m x m)
		true_positives = nn.equal(label_targets, label_predictions) * unlabeled_targets
		correct_label_tokens = nn.equal(label_targets, label_predictions) * unlabeled_targets
		# (n x m x m) -> ()
		n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
		n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
		n_true_positives = tf.reduce_sum(true_positives)
		n_correct_label_tokens = tf.reduce_sum(correct_label_tokens)
		# () - () -> ()
		n_false_positives = n_unlabeled_predictions - n_true_positives
		n_false_negatives = n_unlabeled_targets - n_true_positives
		# (n x m x m) -> (n)
		n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2])
		n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
		n_correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=[1, 2])
		# (n) x 2 -> ()
		n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
		n_correct_label_sequences = tf.reduce_sum(
			nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))

		# -----------------------------------------------------------
		# Populate the output dictionary
		# rho = self.loss_interpolation

		# =================debug======================================

		# =================debug======================================
		second_outputs = {}

		second_outputs['unlabeled_targets'] = unlabeled_targets
		second_outputs['unlabeled_predictions'] = unlabeled_predictions
		second_outputs['unlabeled_loss'] = tf.stop_gradient(outputs['unlabeled_loss'])
		second_outputs['n_unlabeled_true_positives'] = tf.stop_gradient(outputs['n_unlabeled_true_positives'])
		second_outputs['n_unlabeled_false_positives'] = tf.stop_gradient(outputs['n_unlabeled_false_positives'])
		second_outputs['n_unlabeled_false_negatives'] = tf.stop_gradient(outputs['n_unlabeled_false_negatives'])
		second_outputs['n_correct_unlabeled_sequences'] = tf.stop_gradient(outputs['n_correct_unlabeled_sequences'])
		if not self.factorized:
			unlabeled_true_positives = unlabeled_predictions * unlabeled_targets
			n_unlabeled_true_positives = tf.reduce_sum(unlabeled_true_positives)
			n_unlabeled_false_positives = n_unlabeled_predictions - n_unlabeled_true_positives
			n_unlabeled_false_negatives = n_unlabeled_targets - n_unlabeled_true_positives
			# n_correct_unlabeled_sequences = tf.reduce_sum(
			# 	nn.equal(n_unlabeled_true_positives_per_sequence, n_targets_per_sequence))
			second_outputs['n_unlabeled_true_positives'] = n_unlabeled_true_positives
			second_outputs['n_unlabeled_false_positives'] = n_unlabeled_false_positives
			second_outputs['n_unlabeled_false_negatives'] = n_unlabeled_false_negatives
			# second_outputs['n_correct_unlabeled_sequences'] = n_correct_unlabeled_sequences

		second_outputs['label_targets'] = label_targets
		second_outputs['unary_predictions'] = unary_predictions
		second_outputs['label_predictions'] = label_predictions
		if self.factorized:
			second_outputs['probabilities'] = label_probabilities * head_probabilities
		else:
			second_outputs['probabilities'] = label_probabilities
		second_outputs['label_probabilities'] = label_probabilities
		second_outputs['label_loss'] = label_loss
		second_outputs['loss'] = label_loss
		# Combination of labeled loss and unlabeled loss
		# outputs['loss'] = 2 * ((1 - rho) * outputs['loss'] + rho * label_loss)

		second_outputs['n_true_positives'] = n_true_positives
		second_outputs['n_false_positives'] = n_false_positives
		second_outputs['n_false_negatives'] = n_false_negatives
		second_outputs['n_correct_sequences'] = n_correct_sequences
		second_outputs['n_correct_label_tokens'] = n_correct_label_tokens
		second_outputs['n_correct_label_sequences'] = n_correct_label_sequences
		# pdb.set_trace()

		second_outputs['transposed_logits'] = transposed_logits
		second_outputs['tokens'] = token_weights3D
		return second_outputs

	# =============================================================
	def _count(self, node):
		if node not in ('_', ''):
			node = node.split('|')
			for edge in node:
				edge = edge.split(':', 1)
				head, rel = edge
				self.counts[rel] += 1
		return

	# =============================================================
	def add(self, token):
		""""""

		return self.index(token)

	# =============================================================
	# token should be: 1:rel|2:acl|5:dep
	def index(self, token):
		""""""

		nodes = []
		if token != '_':
			token = token.split('|')
			for edge in token:
				head, semrel = edge.split(':', 1)
				nodes.append((int(head), super(SecondGraphTokenVocab, self).__getitem__(semrel)))
		return nodes

	# =============================================================
	# index should be [(1, 12), (2, 4), (5, 2)]
	def token(self, index):
		""""""

		nodes = []
		for (head, semrel) in index:
			nodes.append('{}:{}'.format(head, super(SecondGraphTokenVocab, self).__getitem__(semrel)))
		return '|'.join(nodes)

	# =============================================================
	def get_root(self):
		""""""

		return '_'

	# =============================================================
	def __getitem__(self, key):
		if isinstance(key, six.string_types):
			nodes = []
			if key != '_':
				token = key.split('|')
				for edge in token:
					head, rel = edge.split(':', 1)
					nodes.append((int(head), super(SecondGraphTokenVocab, self).__getitem__(rel)))
			return nodes
		elif hasattr(key, '__iter__'):
			if len(key) > 0 and hasattr(key[0], '__iter__'):
				if len(key[0]) > 0 and isinstance(key[0][0], six.integer_types + (np.int32, np.int64)):
					nodes = []
					for (head, rel) in key:
						nodes.append('{}:{}'.format(head, super(SecondGraphTokenVocab, self).__getitem__(rel)))
					return '|'.join(nodes)
				else:
					return [self[k] for k in key]
			else:
				return '_'
		else:
			raise ValueError('key to GraphTokenVocab.__getitem__ must be (iterable of) strings or iterable of integers')

	@property
	def num_iteration(self):
		return self._config.getint(self, 'num_iteration')

	@property
	def unary_hidden_size(self):
		return self._config.getint(self, 'unary_hidden_size')

	@property
	def sib_hidden_size(self):
		return self._config.getint(self, 'sib_hidden_size')

	@property
	def use_sib(self):
		return self._config.getboolean(self, 'use_sib')

	@property
	def transposed(self):
		return self._config.getboolean(self, 'transposed')

#***************************************************************
class GraphLabelVocab(GraphTokenVocab):
	""""""

	_depth = -1

	#=============================================================
	def __init__(self, *args, **kwargs):
		super(GraphLabelVocab, self).__init__(*args, **kwargs)
		return
	#=============================================================
	def get_bilinear_classifier(self, layer, unlabeled_targets, token_weights, variable_scope=None, reuse=False, debug=False):
		""""""
		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		add_linear = self.add_linear
		with tf.variable_scope(variable_scope or self.field):
			for i in six.moves.range(0, self.n_layers-1):
				with tf.variable_scope('FC-%d' % i):
					layer = classifiers.hidden(layer, 2*self.hidden_size,
																			hidden_func=self.hidden_func,
																			hidden_keep_prob=hidden_keep_prob)
			with tf.variable_scope('FC-top'):
				layers = classifiers.hiddens(layer, 2*[self.hidden_size],
																		hidden_func=self.hidden_func,
																		hidden_keep_prob=hidden_keep_prob)
			layer1, layer2 = layers.pop(0), layers.pop(0)

			with tf.variable_scope('Classifier'):
				if self.diagonal:
					label_layer = classifiers.diagonal_bilinear_layer(
						layer1, layer2,
						hidden_keep_prob=hidden_keep_prob,
						add_linear=False)
			with tf.variable_scope('Label_Classifier'):
				# (n x m x m x d) -> (n x m x m x c)
				logits = classifiers.hidden(label_layer, len(self), hidden_func=nonlin.identity,hidden_keep_prob=1)

		#-----------------------------------------------------------
		# Process the targets
		# (n x m x m)
		label_targets = self.placeholder
		#-----------------------------------------------------------
		# Process the logits
		transposed_logits = logits

		#-----------------------------------------------------------
		# Compute the probabilities/cross entropy
		# (n x m x m x c) -> (n x m x m x c)
		label_probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
		# (n x m x m), (n x m x m x c), (n x m x m) -> ()
		label_loss = tf.losses.sparse_softmax_cross_entropy(label_targets, transposed_logits, weights=token_weights*unlabeled_targets)
		#-----------------------------------------------------------
		# Populate the output dictionary
		outputs={}
		#rho = self.loss_interpolation
		if self.use_embedding:
			print('use embedding label information')
			word_embeddings = tf.get_variable("word_embeddings", [len(self), self.hidden_size], initializer=tf.random_normal_initializer(stddev=0.01))
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(word_embeddings))
			label_layer = tf.nn.embedding_lookup(word_embeddings, tf.argmax(transposed_logits, axis=-1, output_type=tf.int32))
		else:
			print('end2end label training')

		outputs['label_layer'] = tf.transpose(label_layer,[0,2,1,3])
		outputs['label_targets'] = label_targets
		outputs['label_predictions'] = tf.argmax(transposed_logits, axis=-1, output_type=tf.int32)
		outputs['label_probabilities'] = label_probabilities
		outputs['label_loss'] = label_loss
		outputs['rho'] = self.loss_interpolation
		return outputs
	@property
	def use_embedding(self):
		try:
			return self._config.getboolean(self,'use_embedding')
		except:
			return False

#***************************************************************
class AttributeVocab(TokenVocab):
	def __init__(self, *args, **kwargs):
		""""""
		#self.vocab_savename='attr-token.list'
		kwargs['placeholder_shape'] = [None, None, None]
		super(AttributeVocab, self).__init__(*args, **kwargs)
		return

	def _count(self, node):
		if node not in ('_', ''):
			node = node.split('|')
			for attr in node:
				self.counts[attr] += 1
		return
		#=============================================================
	def index(self, token):
		""""""

		nodes = []
		if token != '_':
			token = token.split('|')
			# pdb.set_trace()
			for edge in token:
				nodes.append(self._str2idx[edge])
		return nodes

	#=============================================================
	def token(self, index):
		""""""

		nodes = []
		for semrel in index:
			#nodes.append(super(AttributeVocab, self).__getitem__(semrel))
			nodes.append(self._idx2str[semrel])
			#nodes.append('{}:{}'.format(super(AttributeVocab, self).__getitem__(semrel)))
		return '|'.join(nodes)

	def add(self, token):
		""""""
		#pdb.set_trace()
		res=np.zeros(len(self),dtype=np.int32)
		res[self.index(token)]=1
		return res
	#=============================================================
	def get_root(self):
		""""""

		return '_'

	#=============================================================
	def __getitem__(self, key):
		res=[]
		for sent in key:
			nodes=[]
			for attr_list in sent:
				if len(attr_list)>0:
					node=[]
					for attr in attr_list:
						node.append(self._idx2str[attr])
					nodes.append('|'.join(node))
				else:
					nodes.append('_')
			res.append(nodes)
		return res
		#=============================================================
	def get_bilinear_classifier(self, layer, outputs, token_weights, variable_scope=None, reuse=False, debug=False):
		""""""
		#pdb.set_trace()
		recur_layer = layer
		hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
		with tf.variable_scope(variable_scope or self.field):
			with tf.variable_scope('FC-top'):
				layer = classifiers.hidden(layer, self.hidden_size,
																		hidden_func=self.hidden_func,
																		hidden_keep_prob=hidden_keep_prob)


			with tf.variable_scope('Classifier'):
					logits = classifiers.linear_classifier(layer, len(self),hidden_keep_prob=hidden_keep_prob)

		#-----------------------------------------------------------
		# Process the targets
		# (n x m x c)
		attr_targets = self.placeholder
		# unlabeled_predictions = outputs['unlabeled_predictions']
		# unlabeled_targets = outputs['unlabeled_targets']

		#-----------------------------------------------------------
		# Process the logits
		# (n x m x c)
		transposed_logits = logits

		#-----------------------------------------------------------
		# Compute the probabilities/cross entropy
		probabilities = tf.nn.sigmoid(logits) * tf.to_float(token_weights)
		# (n x m x m), (n x m x m), (n x m x m) -> ()
		loss = tf.losses.sigmoid_cross_entropy(attr_targets, transposed_logits, weights=token_weights)

		#-----------------------------------------------------------
		# Compute the predictions/accuracy
		predictions = nn.greater(logits, 0, dtype=tf.int32) * token_weights

		# (n x m x m) (*) (n x m x m) -> (n x m x m)
		true_positives = predictions * attr_targets
		# (n x m x m) -> ()
		n_predictions = tf.reduce_sum(predictions)
		n_targets = tf.reduce_sum(attr_targets)
		n_true_positives = tf.reduce_sum(true_positives)
		# () - () -> ()
		n_false_positives = n_predictions - n_true_positives
		n_false_negatives = n_targets - n_true_positives
		# # (n x m x m) -> (n)
		n_targets_per_sequence = tf.reduce_sum(attr_targets, axis=[1,2])
		n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1,2])
		n_targets_per_token = tf.reduce_sum(attr_targets, axis=[2])
		n_true_positives_per_token = tf.reduce_sum(true_positives, axis=[2])
		#n_correct_tokens = tf.reduce_sum(nn.equal(n_targets_per_token, n_true_positives_per_token))
		# # (n) x 2 -> ()
		n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

		#-----------------------------------------------------------
		# Populate the output dictionary
		rho = self.loss_interpolation
		unlabeled_loss=outputs['unlabeled_loss']
		label_loss=outputs['label_loss']
		outputs={}
		outputs['attribute_loss'] = loss
		# Combination of labeled loss and unlabeled loss
		outputs['loss'] = self.loss_attr_interpolation * loss

		outputs['probabilities'] = probabilities
		outputs['attribute_prediction'] = predictions
		outputs['n_true_positives'] = n_true_positives
		outputs['n_false_positives'] = n_false_positives
		outputs['n_false_negatives'] = n_false_negatives
		outputs['n_correct_sequences'] = n_correct_sequences
		outputs['n_correct_tokens'] = n_true_positives

		if debug:
			outputs['target'] = attr_targets
			outputs['n_targets_per_token']=n_targets_per_token
			outputs['n_true_positives_per_token']=n_true_positives_per_token
		# outputs['n_correct_label_sequences'] = n_correct_label_sequences
		return outputs

	@property
	def loss_attr_interpolation(self):
		return self._config.getfloat(self, 'loss_attr_interpolation')

#--------------------------------------------------------------------

#--------------------------------------------------------------------------

#***************************************************************
class FormTokenVocab(TokenVocab, cv.FormVocab):
	pass
class LemmaTokenVocab(TokenVocab, cv.LemmaVocab):
	pass
class UPOSTokenVocab(TokenVocab, cv.UPOSVocab):
	pass
class XPOSTokenVocab(TokenVocab, cv.XPOSVocab):
	pass
class DeprelTokenVocab(TokenVocab, cv.DeprelVocab):
	pass
class DepheadTokenVocab(TokenVocab, cv.DepheadVocab):
	pass
class SemrelGraphTokenVocab(GraphTokenVocab, cv.SemrelVocab):
	pass
class SpanendGraphTokenVocab(GraphTokenVocab, cv.SpanrelVocab):
	pass
#-----------------------------------------------------------
class SecondOrderGraphTokenVocab(SecondGraphTokenVocab, cv.SemrelVocab):
  pass
#-----------------------------------------------------------
class SemrelGraphLabelVocab(GraphLabelVocab, cv.SemrelVocab):
	pass

class AttributeTokenVocab(AttributeVocab, cv.AttrVocab):
	pass
#=========================================================
class FlagTokenVocab(TokenVocab, cv.FlagVocab):
	pass
class PredicateTokenVocab(TokenVocab, cv.PredicateVocab):
	pass
class SpanrelGraphTokenVocab(SpanTokenVocab, cv.SpanrelVocab):
	pass
#=========================================================