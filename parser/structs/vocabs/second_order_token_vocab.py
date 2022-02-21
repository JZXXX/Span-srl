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
#***************************************************************
class SecondOrderTokenVocab(CountVocab):
  """"""
  
  _save_str = 'tokens'
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(SecondOrderTokenVocab, self).__init__(*args, **kwargs)
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
    #pdb.set_trace()
    for train_conllu in train_conllus:
      with codecs.open(train_conllu, encoding='utf-8', errors='ignore') as f:
        for line in f:
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
  #==========================3.1============================
  @property
  def unary_hidden_size(self):
    return self._config.getint(self, 'unary_hidden_size')
  @property
  def sib_hidden_size(self):
    return self._config.getint(self, 'sib_hidden_size')
  #==========================3.1============================
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
  def num_iteration(self):
    return self._config.getfloat(self, 'num_iteration')
  @property
  def self_minus(self):
    return self._config.getboolean(self,'self_minus')
  @property
  def use_sib(self):
    return self._config.getboolean(self,'use_sib')
  @property
  def use_gp(self):
    return self._config.getboolean(self,'use_gp')
  @property
  def use_cop(self):
    return self._config.getboolean(self,'use_cop')
  @property
  def transposed(self):
    return self._config.getboolean(self,'transposed')
#***************************************************************
class GraphSecondTokenVocab(SecondOrderTokenVocab):
  """"""
  
  _depth = -1
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    kwargs['placeholder_shape'] = [None, None, None]
    super(GraphSecondTokenVocab, self).__init__(*args, **kwargs)
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
  def get_bilinear_classifier(self, layer, outputs, token_weights, variable_scope=None, reuse=False, debug=False,token_weights4D=None):
    """"""
    #pdb.set_trace()
    recur_layer = layer
    hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
    add_linear = self.add_linear
    hidden_sizes = 2*self.unary_hidden_size + 2*self.sib_hidden_size # sib_head, sib_dep
    with tf.variable_scope(variable_scope or self.field):
      for i in six.moves.range(0, self.n_layers-1):
        with tf.variable_scope('FC-%d' % i):
          layer = classifiers.hidden(layer, hidden_sizes,
                                      hidden_func=self.hidden_func,
                                      hidden_keep_prob=hidden_keep_prob)
      with tf.variable_scope('FC-top'):
        layers = classifiers.hiddens(layer, 2*[self.unary_hidden_size]+2*[self.sib_hidden_size],
                                    hidden_func=self.hidden_func,
                                    hidden_keep_prob=hidden_keep_prob)
      unary_head, unary_dep, sib_head, sib_dep = layers.pop(0), layers.pop(0), layers.pop(0), layers.pop(0)
      
      with tf.variable_scope('Classifier'):
        if self.diagonal:
          # (n x ma x o x mb)
          unary_logits = classifiers.diagonal_bilinear_classifier(
            unary_head, unary_dep, len(self),
            hidden_keep_prob=hidden_keep_prob,
            add_linear=add_linear)
          if self.use_sib:
            with tf.variable_scope('Sibling'):
              # layer = (n x o x o x ma x mb x mc)
              sib_logits = classifiers.trilinear_classifier(
                sib_head, sib_dep, sib_dep, len(self),
                hidden_keep_prob=hidden_keep_prob,
                add_linear=add_linear)
        else:
          #(n x ma x o x mb)
          unary_logits = classifiers.bilinear_classifier(
            unary_head, unary_dep, len(self),
            hidden_keep_prob=hidden_keep_prob,
            add_linear=add_linear)
          if self.use_sib:
            with tf.variable_scope('Sibling'):
              # layer = (n x o x o x ma x mb x mc) 
              sib_logits = classifiers.trilinear_classifier(
                sib_head, sib_dep, sib_dep, len(self),
                hidden_keep_prob=hidden_keep_prob,
                add_linear=add_linear)
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

      #(n x ma x o x mb) -> (n x o x ma x mb)
      unary_logits=tf.transpose(unary_logits,perm=[0,2,1,3])
      # unary_potential=-unary
      q_value=unary_logits
      # o is label of Q(a,b), i is label of Q(a,c)
      #1 sibling (n x o x i x ma x mb x mc) * (n x i x ma x mc) -> (n x o x i x ma x mb)
      #2 grand parent (n x o x i x ma x mb x mc) * (n x i x mb x mc) -> (n x o x i x ma x mb)
      #3 coparent (n x o x i x ma x mb x mc) * (n x i x mc x mb) -> (n x o x i x ma x mb)

      #binary_shape = nn.get_sizes(layer_sib)
      if debug:
        outputs['q_value_orig']=q_value
      for i in range(int(self.num_iteration)):
        q_value=tf.nn.softmax(q_value,1)
        if debug and i==0:
          outputs['q_value_old']=q_value
        # q_value (n x o x mp x mc) * sib_logits (n x o x o x mp x ma x mb) -> F_temp (n x o x mp x ma x mb)
        if self.use_sib:
          # second_temp_sib = tf.einsum('niac,noiabc->noiab', q_value, layer_sib)
          F_temp_sib = tf.einsum('norpik,nrpk->nopik', sib_logits, unary_logits)
          # n x mp x o x ma x mb
          F_temp_sib = tf.transpose(F_temp_sib,perm=[0,2,1,3,4])
          F_temp_upper = (F_temp_sib - tf.linalg.band_part(F_temp_sib, 0, -1)) + tf.transpose((F_temp_sib - tf.linalg.band_part(F_temp_sib, -1, 0)),
                                                                      perm=[0,1,2,4,3])
          F_temp_lower = (F_temp_sib - tf.linalg.band_part(F_temp_sib, -1, 0)) + tf.transpose((F_temp_sib - tf.linalg.band_part(F_temp_sib, 0, -1)),
                                                                      perm=[0,1,2,4,3])
          F_temp_upper = tf.transpose(F_temp_upper, perm=[0,1,3,4,2])
          F_temp_lower = tf.transpose(F_temp_lower, perm=[0,1,3,4,2])
          # n x mp x ma x mb x o : a->i; b->k
          F_temp_sib = F_temp_upper + F_temp_lower
          # ->root 0 ; ->pad-> 0
          # F_temp_sib = F_temp_sib * token_weights4D
          # n x o x mp x ma x mb
          F_temp_sib = tf.transpose(F_temp_sib, perm=[0,4,1,2,3])

        else:
          F_temp_sib=0
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
        #(n x o x mp x mh1 x mh2)
        # second_temp = second_temp_sib + second_temp_gp + second_temp_gp2 + second_temp_cop
        F = F_temp_sib

        #Second order potential update function
        #second_temp=unary_potential - second_temp
        # (n x o x mp x mh)
        q_value=unary_logits + tf.reduce_sum(F,-1)
        #q_value=tf.stack([unary_potential[:,:,0,:],second_temp],axis=2)
        if debug:
          outputs['q_value'+str(i)]=q_value      



    #-----------------------------------------------------------
    # Process the targets
    # (n x m x m)
    label_targets = self.placeholder
    unlabeled_predictions = outputs['unlabeled_predictions']
    unlabeled_targets = outputs['unlabeled_targets']
    if debug:
      outputs['labelprint']={}
      outputs['labelprint']['q_value']=q_value
      outputs['labelprint']['unary']=unary_logits
      #outputs['printdata']['binary']=binary
      outputs['labelprint']['second_temp']=F
      if self.use_sib:
        outputs['labelprint']['second_temp_sib']=F_temp_sib
        outputs['labelprint']['layer_sib']=sib_logits
      # if self.use_gp:
      #   #outputs['printdata']['second_temp_gp']=second_temp_gp
      #   outputs['labelprint']['second_temp_gp']=second_temp_gp
      #   outputs['labelprint']['second_temp_gp2']=second_temp_gp2
      #   outputs['labelprint']['layer_gp']=layer_gp
      # if self.use_cop:
      #   outputs['labelprint']['layer_cop']=layer_cop
      #   outputs['labelprint']['second_temp_cop']=second_temp_cop
      outputs['labelprint']['unary_head']=unary_head
      outputs['labelprint']['unary_dep']=unary_dep
      outputs['labelprint']['sib_head']=sib_head
      outputs['labelprint']['sib_dep'] = sib_dep
      outputs['labelprint']['targets']=unlabeled_targets
      outputs['labelprint']['token_weights']=token_weights
    #-----------------------------------------------------------
    # Process the logits
    # (n x m x c x m) -> (n x m x m x c)
    #transposed_logits = tf.transpose(logits, [0,1,3,2])
    #q_value=unary
    if self.transposed:
      # need transpose as the target is the transpose
      # (n x o x ma x mb) -> (n x mb x ma x o)
      transposed_logits = tf.transpose(q_value,[0,3,2,1])
    else:
      # (n x o x ma x mb) -> (n x ma x mb x o)
      transposed_logits = tf.transpose(q_value,[0,2,3,1])
    
    #-----------------------------------------------------------
    # Compute the probabilities/cross entropy
    # (n x m x m) -> (n x m x m x 1)
    head_probabilities = tf.expand_dims(tf.stop_gradient(outputs['probabilities']), axis=-1)
    # (n x m x m x c) -> (n x m x m x c)
    label_probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights, axis=-1))
    # (n x m x m), (n x m x m x c), (n x m x m) -> ()
    label_loss = tf.losses.sparse_softmax_cross_entropy(label_targets, transposed_logits, weights=token_weights*unlabeled_targets)
    
    #-----------------------------------------------------------
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
    n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1,2])
    n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1,2])
    n_correct_label_tokens_per_sequence = tf.reduce_sum(correct_label_tokens, axis=[1,2])
    # (n) x 2 -> ()
    n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
    n_correct_label_sequences = tf.reduce_sum(nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))
    
    #-----------------------------------------------------------
    # Populate the output dictionary
    rho = self.loss_interpolation
    outputs['label_targets'] = label_targets
    outputs['probabilities'] = label_probabilities * head_probabilities
    outputs['label_loss'] = label_loss
    # Combination of labeled loss and unlabeled loss
    outputs['loss'] = 2*((1-rho) * outputs['loss'] + rho * label_loss)
    
    outputs['n_true_positives'] = n_true_positives
    outputs['n_false_positives'] = n_false_positives
    outputs['n_false_negatives'] = n_false_negatives
    outputs['n_correct_sequences'] = n_correct_sequences
    outputs['n_correct_label_tokens'] = n_correct_label_tokens
    outputs['n_correct_label_sequences'] = n_correct_label_sequences
    return outputs
  
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
      with tf.variable_scope('FC-top' % i):
        layers = classifiers.hidden(layer, 2*[self.hidden_size],
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
        # (n x m x m) (*) (n x m x m) -> (n x m x m)
        unlabeled_true_positives = unlabeled_predictions * unlabeled_targets
        true_positives = nn.equal(targets, predictions) * unlabeled_true_positives
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
    return outputs
    
  #=============================================================
  def _count(self, node):
    #pdb.set_trace()
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
      #pdb.set_trace()
      token = token.split('|')
      for edge in token:
        head, semrel = edge.split(':', 1)
        nodes.append( (int(head), super(GraphSecondTokenVocab, self).__getitem__(semrel)) )
    return nodes
  
  #=============================================================
  # index should be [(1, 12), (2, 4), (5, 2)]
  def token(self, index):
    """"""
    
    nodes = []
    for (head, semrel) in index:
      nodes.append('{}:{}'.format(head, super(GraphSecondTokenVocab, self).__getitem__(semrel)))
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
          nodes.append( (int(head), super(GraphSecondTokenVocab, self).__getitem__(rel)) )
      return nodes
    elif hasattr(key, '__iter__'):
      if len(key) > 0 and hasattr(key[0], '__iter__'):
        if len(key[0]) > 0 and isinstance(key[0][0], six.integer_types + (np.int32, np.int64)):
          nodes = []
          for (head, rel) in key:
            nodes.append('{}:{}'.format(head, super(GraphSecondTokenVocab, self).__getitem__(rel)))
          return '|'.join(nodes)
        else:
          return [self[k] for k in key]
      else:
        return '_'
    else:
      raise ValueError('key to GraphSecondTokenVocab.__getitem__ must be (iterable of) strings or iterable of integers')

#***************************************************************
class SecondOrderGraphTokenVocab(GraphSecondTokenVocab, cv.SemrelVocab):
  pass
