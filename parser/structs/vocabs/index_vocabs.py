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

from .base_vocabs import BaseVocab
from . import conllu_vocabs as cv

from parser.neural import nn, nonlin, classifiers, embeddings
import pdb
#***************************************************************
class IndexVocab(BaseVocab):
    """"""

    #=============================================================
    def __init__(self, *args, **kwargs):
        """"""

        super(IndexVocab, self).__init__(*args, **kwargs)

        self.PAD_STR = '_'
        self.PAD_IDX = -1
        self.ROOT_STR = '0'
        self.ROOT_IDX = 0
        self.BOS_STR = '<bos>'
        self.BOS_IDX = 3
        self.EOS_STR = '<eos>'
        self.EOS_IDX = 4
        return

    #=============================================================
    def add(self, token):
        """"""

        return self.index(token)

    #=============================================================
    def token(self, index):
        """"""

        if index > -1:
            return str(index)
        else:
            return '_'

    #=============================================================
    def index(self, token):
        """"""

        if token != '_':
            return int(token)
        elif token == self.BOS_STR:
            return self.BOS_IDX
        elif token == self.EOS_STR:
            return self.EOS_IDX
        else:
            return -1

    #=============================================================
    def get_root(self):
        """"""

        return self.ROOT_STR

    #=============================================================
    def get_bos(self):
        """"""

        return self.BOS_STR

    #=============================================================
    def get_eos(self):
        """"""

        return self.EOS_STR

    #=============================================================
    def get_bilinear_classifier(self, layer, token_weights, variable_scope=None, reuse=False):
        """"""

        recur_layer = layer
        hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
        hidden_func = self.hidden_func
        hidden_size = self.hidden_size
        add_linear = self.add_linear
        linearize = self.linearize
        distance = self.distance
        n_splits = 2*(1+linearize+distance)
        with tf.variable_scope(variable_scope or self.field):
            for i in six.moves.range(0, self.n_layers-1):
                with tf.variable_scope('FC-%d' % i):
                    layer = classifiers.hidden(layer, n_splits*hidden_size,
                                                                        hidden_func=hidden_func,
                                                                        hidden_keep_prob=hidden_keep_prob)
            with tf.variable_scope('FC-top'):
                layers = classifiers.hiddens(layer, n_splits*[hidden_size],
                                                                        hidden_func=hidden_func,
                                                                        hidden_keep_prob=hidden_keep_prob)
            layer1, layer2 = layers.pop(0), layers.pop(0)
            if linearize:
                lin_layer1, lin_layer2 = layers.pop(0), layers.pop(0)
            if distance:
                dist_layer1, dist_layer2 = layers.pop(0), layers.pop(0)

            with tf.variable_scope('Attention'):
                if self.diagonal:
                    logits, _ = classifiers.diagonal_bilinear_attention(
                        layer1, layer2,
                        hidden_keep_prob=hidden_keep_prob,
                        add_linear=add_linear)
                    if linearize:
                        with tf.variable_scope('Linearization'):
                            lin_logits = classifiers.diagonal_bilinear_discriminator(
                                lin_layer1, lin_layer2,
                                hidden_keep_prob=hidden_keep_prob,
                                add_linear=add_linear)
                    if distance:
                        with tf.variable_scope('Distance'):
                            dist_lamda = 1+tf.nn.softplus(classifiers.diagonal_bilinear_discriminator(
                                dist_layer1, dist_layer2,
                                hidden_keep_prob=hidden_keep_prob,
                                add_linear=add_linear))
                else:
                    logits, _ = classifiers.bilinear_attention(
                        layer1, layer2,
                        hidden_keep_prob=hidden_keep_prob,
                        add_linear=add_linear)
                    if linearize:
                        with tf.variable_scope('Linearization'):
                            lin_logits = classifiers.bilinear_discriminator(
                                lin_layer1, lin_layer2,
                                hidden_keep_prob=hidden_keep_prob,
                                add_linear=add_linear)
                    if distance:
                        with tf.variable_scope('Distance'):
                            dist_lamda = 1+tf.nn.softplus(classifiers.bilinear_discriminator(
                                dist_layer1, dist_layer2,
                                hidden_keep_prob=hidden_keep_prob,
                                add_linear=add_linear))

                #-----------------------------------------------------------
                # Process the targets
                targets = self.placeholder
                shape = tf.shape(layer1)
                batch_size, bucket_size = shape[0], shape[1]
                # (1 x m)
                ids = tf.expand_dims(tf.range(bucket_size), 0)
                # (1 x m) -> (1 x 1 x m)
                head_ids = tf.expand_dims(ids, -2)
                # (1 x m) -> (1 x m x 1)
                dep_ids = tf.expand_dims(ids, -1)
                if linearize:
                    # Wherever the head is to the left
                    # (n x m), (1 x m) -> (n x m)
                    lin_targets = tf.to_float(tf.less(targets, ids))
                    # cross-entropy of the linearization of each i,j pair
                    # (1 x 1 x m), (1 x m x 1) -> (n x m x m)
                    lin_ids = tf.tile(tf.less(head_ids, dep_ids), [batch_size, 1, 1])
                    # (n x 1 x m), (n x m x 1) -> (n x m x m)
                    lin_xent = -tf.nn.softplus(tf.where(lin_ids, -lin_logits, lin_logits))
                    # add the cross-entropy to the logits
                    # (n x m x m), (n x m x m) -> (n x m x m)
                    logits += tf.stop_gradient(lin_xent)
                if distance:
                    # (n x m) - (1 x m) -> (n x m)
                    dist_targets = tf.abs(targets - ids)
                    # KL-divergence of the distance of each i,j pair
                    # (1 x 1 x m) - (1 x m x 1) -> (n x m x m)
                    dist_ids = tf.to_float(tf.tile(tf.abs(head_ids - dep_ids), [batch_size, 1, 1]))+1e-12
                    # (n x m x m), (n x m x m) -> (n x m x m)
                    #dist_kld = (dist_ids * tf.log(dist_lamda / dist_ids) + dist_ids - dist_lamda)
                    dist_kld = -tf.log((dist_ids - dist_lamda)**2/2 + 1)
                    # add the KL-divergence to the logits
                    # (n x m x m), (n x m x m) -> (n x m x m)
                    logits += tf.stop_gradient(dist_kld)

                #-----------------------------------------------------------
                # Compute probabilities/cross entropy
                # (n x m) + (m) -> (n x m)
                non_pads = tf.to_float(token_weights) + tf.to_float(tf.logical_not(tf.cast(tf.range(bucket_size), dtype=tf.bool)))
                # (n x m x m) o (n x 1 x m) -> (n x m x m)
                probabilities = tf.nn.softmax(logits) * tf.expand_dims(non_pads, -2)
                # (n x m), (n x m x m), (n x m) -> ()
                loss = tf.losses.sparse_softmax_cross_entropy(
                    targets,
                    logits,
                    weights=token_weights)
                # (n x m) -> (n x m x m x 1)
                one_hot_targets = tf.expand_dims(tf.one_hot(targets, bucket_size), -1)
                # (n x m) -> ()
                n_tokens = tf.to_float(tf.reduce_sum(token_weights))
                if linearize:
                    # (n x m x m) -> (n x m x 1 x m)
                    lin_xent_reshaped = tf.expand_dims(lin_xent, -2)
                    # (n x m x 1 x m) * (n x m x m x 1) -> (n x m x 1 x 1)
                    lin_target_xent = tf.matmul(lin_xent_reshaped, one_hot_targets)
                    # (n x m x 1 x 1) -> (n x m)
                    lin_target_xent = tf.squeeze(lin_target_xent, [-1, -2])
                    # (n x m), (n x m), (n x m) -> ()
                    loss -= tf.reduce_sum(lin_target_xent*tf.to_float(token_weights)) / (n_tokens + 1e-12)
                if distance:
                    # (n x m x m) -> (n x m x 1 x m)
                    dist_kld_reshaped = tf.expand_dims(dist_kld, -2)
                    # (n x m x 1 x m) * (n x m x m x 1) -> (n x m x 1 x 1)
                    dist_target_kld = tf.matmul(dist_kld_reshaped, one_hot_targets)
                    # (n x m x 1 x 1) -> (n x m)
                    dist_target_kld = tf.squeeze(dist_target_kld, [-1, -2])
                    # (n x m), (n x m), (n x m) -> ()
                    loss -= tf.reduce_sum(dist_target_kld*tf.to_float(token_weights)) / (n_tokens + 1e-12)

                #-----------------------------------------------------------
                # Compute predictions/accuracy
                # (n x m x m) -> (n x m)
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
        outputs['unlabeled_targets'] = self.placeholder
        outputs['probabilities'] = probabilities
        outputs['unlabeled_loss'] = loss
        outputs['loss'] = loss

        outputs['unlabeled_predictions'] = predictions
        outputs['predictions'] = predictions
        outputs['correct_unlabeled_tokens'] = correct_tokens
        outputs['n_correct_unlabeled_tokens'] = tf.reduce_sum(correct_tokens)
        outputs['n_correct_unlabeled_sequences'] = tf.reduce_sum(correct_sequences)
        outputs['n_correct_tokens'] = tf.reduce_sum(correct_tokens)
        outputs['n_correct_sequences'] = tf.reduce_sum(correct_sequences)
        return outputs

    #=============================================================
    def __getitem__(self, key):
        if isinstance(key, six.string_types):
            if key == '_':
                return -1
            else:
                return int(key)
        elif isinstance(key, six.integer_types + (np.int32, np.int64)):
            if key > -1:
                return str(key)
            else:
                return '_'
        elif hasattr(key, '__iter__'):
            return [self[k] for k in key]
        else:
            raise ValueError('key to IndexVocab.__getitem__ must be (iterable of) string or integer')
        return
    #=============================================================
    @property
    def distance(self):
        return self._config.getboolean(self, 'distance')
    @property
    def linearize(self):
        return self._config.getboolean(self, 'linearize')
    @property
    def decomposition_level(self):
        return self._config.getint(self, 'decomposition_level')
    @property
    def diagonal(self):
        return self._config.getboolean(self, 'diagonal')
    @property
    def add_linear(self):
        return self._config.getboolean(self, 'add_linear')
    @property
    def n_layers(self):
        return self._config.getint(self, 'n_layers')
    @property
    def hidden_size(self):
        return self._config.getint(self, 'hidden_size')
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


#***************************************************************
class PredPadVocab(IndexVocab):
    # =============================================================
    def __init__(self, *args, **kwargs):
        kwargs['placeholder_shape'] = [None, None]
        super(PredPadVocab, self).__init__(*args, **kwargs)
        return

    # =============================================================
    # token should be: 1:(1,2)-rel|2:(4,6)-acl|5:(8,8)-dep or 1|2|5
    def index(self, token):
        """"""

        nodes = []
        if token != '_':
            token = token.split('|')
            for edge in token:
                head = edge.split(':')[0]
                span_end = edge.split('-', 1)[0].split(':')[1].split(',')[1][:-1]
                nodes.append((int(head), int(span_end)))
        return nodes

    # =============================================================
    # index should be [[1,3], [2,4], [5,3]]
    def token(self, index):
        """"""

        return ['(' + str(head[0]) + ',' + str(head[1]) + ')' for head in index]

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
                    head = edge.split(':')[0]
                    span_end = edge.split('-', 1)[0].split(':')[1].split(',')[1][:-1]
                    nodes.append((int(head), int(span_end)))
            return nodes
        elif hasattr(key, '__iter__'):
            if len(key) > 0:
                if len(key[0]) > 0 and isinstance(key[0][0], six.integer_types + (np.int32, np.int64)):
                    return '|'.join(['(' + str(head[0]) + ',' + str(head[1]) + ')' for head in key])
                else:
                    return [self[k] for k in key]
            else:
                return '_'
        else:
            raise ValueError(
                'Key to SpanIndexVocab.__getitem__ must be (iterable of) strings or iterable of integers list')


#***************************************************************
class SpanIndexVocab(IndexVocab):

    _depth = -4

    #=============================================================
    def __init__(self, *args, **kwargs):
        kwargs['placeholder_shape'] = [None, None, None, None]
        super(SpanIndexVocab, self).__init__(*args, **kwargs)
        self.ROOT_STR = '0'
        self.ROOT_IDX = 0
        return

    # =============================================================
    # token should be: 1:(1,2)-rel|2:(4,6)-acl|5:(8,8)-dep or 1|2|5
    def index(self, token):
        """"""

        nodes = []
        if token != '_':
            token = token.split('|')
            for edge in token:
                head = edge.split(':')[0]
                span_end = edge.split('-',1)[0].split(':')[1].split(',')[1][:-1]
                nodes.append((int(head),int(span_end)))
        return nodes

    # =============================================================
    # index should be [[1,3], [2,4], [5,3]]
    def token(self, index):
        """"""

        return ['('+str(head[0])+','+str(head[1])+')' for head in index]

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
                    head = edge.split(':')[0]
                    span_end = edge.split('-', 1)[0].split(':')[1].split(',')[1][:-1]
                    nodes.append((int(head),int(span_end)))
            return nodes
        elif hasattr(key, '__iter__'):
            if len(key) > 0:
                if len(key[0]) > 0 and isinstance(key[0][0], six.integer_types + (np.int32, np.int64)):
                    return '|'.join(['('+str(head[0])+','+str(head[1])+')' for head in key])
                else:
                    return [self[k] for k in key]
            else:
                return '_'
        else:
            raise ValueError(
                'Key to SpanIndexVocab.__getitem__ must be (iterable of) strings or iterable of integers list')

    # =============================================================
    def get_trilinear_discriminator(self, layer, preds, token_weights, variable_scope=None, reuse=False, debug=False):
        """"""
        #pdb.set_trace()
        outputs = {}
        hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
        add_linear = self.add_linear

        batch_size, bucket_size, input_size = nn.get_sizes(layer)
        # pdb.set_trace()
        one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
        layer_preds = tf.matmul(one_hot_pred, layer)
        pred_size = nn.get_sizes(layer_preds)[1]

        layer_args = layer
        layer_arge = layer
        with tf.variable_scope(variable_scope or self.field):
            for i in six.moves.range(0, self.n_layers):
                with tf.variable_scope('ARGS-FC-%d' % i):#here is FNN? did not run
                    layer_args = classifiers.hidden(layer_args, self.hidden_size,
                                                                         hidden_func=self.hidden_func,
                                                                         hidden_keep_prob=hidden_keep_prob)
                with tf.variable_scope('ARGE-FC-%d' % i):#here is FNN? did not run
                    layer_arge = classifiers.hidden(layer_arge, self.hidden_size,
                                                                         hidden_func=self.hidden_func,
                                                                         hidden_keep_prob=hidden_keep_prob)
                with tf.variable_scope('PRED-FC-%d' % i):
                    layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
                                                   hidden_func=self.hidden_func,
                                                   hidden_keep_prob=hidden_keep_prob)

            with tf.variable_scope('Discriminator'):
                # if self.diagonal:
                # 	logits = classifiers.diagonal_bilinear_discriminator(
                # 		layer1, layer2,
                # 		hidden_keep_prob=hidden_keep_prob,
                # 		add_linear=add_linear)
                # else:
                    #only run here
                # n x p x m x m
                logits = classifiers.span_trilinear_discriminator(
                    layer_preds, layer_args, layer_arge,
                    hidden_keep_prob=hidden_keep_prob,
                    add_linear=add_linear)

        # choose tope_k score to loss
        if self.top_k:
            candicate_args_mask = tf.tile(tf.expand_dims(nn.greater(tf.range(bucket_size), 0), 0), [batch_size, 1])
            candicate_args_mask = tf.expand_dims(candicate_args_mask, axis=-1) * tf.expand_dims(candicate_args_mask,
                                                                                                axis=-2)
            candicate_args_mask = tf.tile(tf.expand_dims(candicate_args_mask,1),[1,pred_size,1,1])
            candicate_args_mask = tf.linalg.band_part(candicate_args_mask, 0, -1)
            if self.limit_len:
                limit_len = 30
                candicate_args_mask = tf.cond(bucket_size > tf.constant(limit_len),
                                              lambda: tf.linalg.band_part(candicate_args_mask, 0, limit_len),
                                              lambda: tf.linalg.band_part(candicate_args_mask, 0, -1))

            # n x p x c x 4
            mask_idx = tf.where(candicate_args_mask)
            # n x p x c
            candicate_logits = tf.reshape(tf.gather_nd(logits, mask_idx),[batch_size, pred_size, -1])
            # pdb.set_trace()
            # n x p x k
            _, logits_idx = tf.math.top_k(candicate_logits, k = self.top_k_para*bucket_size)
            # pdb.set_trace()
            candicate_nums = tf.shape(candicate_logits)[-1]

            # n x p x c
            logits_mask = tf.reduce_sum(tf.one_hot(logits_idx, candicate_nums), axis=-2)
            # n x p x k x 4
            sel_logits_idx = tf.where(logits_mask)
            # n x p x k
            logits_sel = tf.reshape(tf.gather_nd(candicate_logits, sel_logits_idx),[batch_size,pred_size,-1])
            # n x p x k x 4
            logits_idx = tf.reshape(tf.gather_nd(tf.reshape(mask_idx,[batch_size,pred_size,-1,4]),sel_logits_idx),[batch_size,pred_size,-1,4])


        #-----------------------------------------------------------
        # Process the targets
        # (n x p x m x m) -> (n x p x m x m)
        #here in fact is a graph, which is m*m representing the connection between each edge
        unlabeled_targets = self.placeholder#ground truth graph, what is self.placeholder?
        if self.top_k:
            unlabeled_targets_sel = tf.reshape(tf.gather_nd(unlabeled_targets, logits_idx),[batch_size, pred_size, -1])

        #-----------------------------------------------------------
        # Compute probabilities/cross entropy
        # (n x p x m x m) -> (n x p x m x m)
        probabilities = tf.nn.sigmoid(logits) * tf.to_float(token_weights)#token weights is sentence length?

        if self.top_k:
            # (n x p x k), (n x p x k) -> ()
            token_weights_sel = tf.reshape(tf.gather_nd(token_weights, logits_idx),[batch_size, pred_size, -1])
            loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets_sel, logits_sel, weights=token_weights_sel)  # here label_smoothing is 0, the sigmoid XE have any effect?
        else:
            # (n x p x m x m), (n x p x m x m), (n x p x m x m) -> ()
            loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets, logits, weights=token_weights)#here label_smoothing is 0, the sigmoid XE have any effect?

        #======================================================
        # try balance cross entropy
        # if self.balance:
        # 	# pdb.set_trace()
        # 	balance_weights = tf.where(tf.equal(unlabeled_targets,1),tf.to_float(tf.ones_like(unlabeled_targets))*tf.constant(2*self.alpha),tf.to_float(tf.ones_like(unlabeled_targets))*tf.constant(2*(1-self.alpha)))
        # 	loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets, logits, weights=tf.to_float(token_weights)*balance_weights)#here label_smoothing is 0, the sigmoid XE have any effect?
        # elif self.dl_loss:
        # 	gamma = self.gamma
        # 	loss = nn.dsc_loss(unlabeled_targets, probabilities, tf.to_float(token_weights), gamma)
        # else:
        # 	loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets, logits, weights=token_weights)#here label_smoothing is 0, the sigmoid XE have any effect?
        #=======================================================


        #-----------------------------------------------------------
        # Compute predictions/accuracy
        # precision/recall
        # if self.top_k:
        # 	# (n x p x k) -> (n x p x k)
        # 	predictions = nn.greater(logits_sel, 0, dtype=tf.int32)  # edge that predicted
        # 	true_positives = predictions * unlabeled_targets_sel
        # 	# (n x p x k) -> ()
        # 	n_predictions = tf.reduce_sum(predictions)
        # 	n_targets = tf.reduce_sum(unlabeled_targets)
        # 	n_true_positives = tf.reduce_sum(true_positives)
        # 	# () - () -> ()
        # 	n_false_positives = n_predictions - n_true_positives
        # 	n_false_negatives = n_targets - n_true_positives
        # 	# (n x p x m x m) -> (n)
        # 	n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])
        # 	n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
        # 	# (n) x 2 -> ()
        # 	n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))
        #
        # else:
        # (n x p x m x m) -> (n x p x m x m)
        predictions = nn.greater(logits, 0, dtype=tf.int32) * token_weights#edge that predicted
        true_positives = predictions * unlabeled_targets
        # (n x p x m x m) -> ()
        n_predictions = tf.reduce_sum(predictions)
        n_targets = tf.reduce_sum(unlabeled_targets)
        n_true_positives = tf.reduce_sum(true_positives)
        # () - () -> ()
        n_false_positives = n_predictions - n_true_positives
        n_false_negatives = n_targets - n_true_positives
        # (n x p x m x m) -> (n)
        n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1,2,3])
        n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1,2,3])
        # (n) x 2 -> ()
        n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

        #-----------------------------------------------------------
        # Populate the output dictionary
        outputs['unlabeled_targets'] = unlabeled_targets
        outputs['head_probabilities'] = probabilities

        outputs['logits'] = logits*tf.to_float(token_weights)
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

        if self.top_k:
            outputs['logits_idx'] = logits_idx
            outputs['candicate_args_mask'] = candicate_args_mask
            outputs['candicate_logits']=candicate_logits
            outputs['logits_mask']=logits_mask
            outputs['candicate_nums']=candicate_nums

            outputs['logits_sel']=logits_sel

            outputs['logits_idx']=logits_idx

            outputs['token_weights']=token_weights

            outputs['unlabeled_targets_sel']=unlabeled_targets_sel

            outputs['sel_logits_idx'] = sel_logits_idx



        return outputs

    # ==============================================================
    def get_bilinear_discriminator(self, layer, preds, token_weights, variable_scope=None, reuse=False):
        """"""
        # pdb.set_trace()
        outputs = {}
        hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
        add_linear = self.add_linear

        batch_size, bucket_size, input_size = nn.get_sizes(layer)

        # pdb.set_trace()
        one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
        layer_preds = tf.matmul(one_hot_pred, layer)
        pred_nums = nn.get_sizes(layer_preds)[1]

        # layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
        #                         tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)
        # # n x m x m x d -> n x k*d
        # top_layer_args = tf.gather_nd(layer_args, top_args_idx)
        # # n x k x d
        # top_layer_args = tf.reshape(top_layer_args, [batch_size, top_args_nums, input_size*2])

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

        limit_len = 70
        candicate_args_mask = tf.tile(tf.expand_dims(nn.greater(tf.range(bucket_size), 0), 0), [batch_size, 1])
        candicate_args_mask = tf.expand_dims(candicate_args_mask, axis=-1) * tf.expand_dims(candicate_args_mask,
                                                                                            axis=-2)

        candicate_args_mask = tf.cond(bucket_size > tf.constant(limit_len),
                                      lambda: tf.linalg.band_part(candicate_args_mask, 0, limit_len),
                                      lambda: tf.linalg.band_part(candicate_args_mask, 0, -1))
        top_args_idx = tf.reshape(tf.where(candicate_args_mask), [batch_size, -1, 3])
        top_args_nums = nn.get_sizes(top_args_idx)[1]
        # pdb.set_trace()
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
            # top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
            # top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])
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
                attention_score_new = mask + attention_score
                attention_weight = tf.math.softmax(attention_score_new, -1)
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

        # elif args_embed_type == ''
        # n x p x m x m
        unlabeled_targets = self.placeholder
        # n x m x m x p
        unlabeled_targets_trans = tf.transpose(unlabeled_targets, [0, 2, 3, 1])
        # n x k*p
        top_unlabeled_targets = tf.gather_nd(unlabeled_targets_trans, top_args_idx)
        # n x p x k
        top_unlabeled_targets = tf.transpose(tf.reshape(top_unlabeled_targets, [batch_size, top_args_nums, pred_nums]),
                                             [0, 2, 1])

        # n x m x m x p
        token_weights_trans = tf.transpose(token_weights, [0, 2, 3, 1])
        # n x k*p
        top_token_weights = tf.gather_nd(token_weights_trans, top_args_idx)
        # n x p x k
        top_token_weights = tf.transpose(tf.reshape(top_token_weights, [batch_size, top_args_nums, pred_nums]),
                                         [0, 2, 1])

        if self.second_order:
            layer_args_sib = top_layer_args  # n x k x d
            layer_preds_sib = layer_preds

            if self.co_parent:
                layer_args_cop = top_layer_args  # n x k x d
                layer_preds_cop = layer_preds

        with tf.variable_scope(variable_scope or self.field):

            for i in six.moves.range(0, self.n_layers):
                with tf.variable_scope('ARGS-FC-%d' % i):  # here is FNN? did not run
                    # n x k x d
                    top_layer_args = classifiers.hidden(top_layer_args, self.hidden_size,
                                                        hidden_func=self.hidden_func,
                                                        hidden_keep_prob=hidden_keep_prob)
                with tf.variable_scope('PRED-FC-%d' % i):
                    # n x p x d
                    layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
                                                     hidden_func=self.hidden_func,
                                                     hidden_keep_prob=hidden_keep_prob)

            with tf.variable_scope('Discriminator'):

                # n x p x k
                logits = classifiers.span_bilinear_discriminator(
                    layer_preds, top_layer_args,
                    hidden_keep_prob=hidden_keep_prob,
                    add_linear=add_linear,
                    target_model=self.target_model,
                    tri_std=self.tri_std_unary)

            if self.second_order:
                hidden_keep_prob_tri = 1 if reuse else self.hidden_keep_prob_tri

                for i in six.moves.range(0, self.n_layers):
                    with tf.variable_scope('SEC-ARG-FC-%d' % i):
                        layer_args_sib = classifiers.hidden(layer_args_sib, self.hidden_size,
                                                            hidden_func=self.hidden_func,
                                                            hidden_keep_prob=hidden_keep_prob)
                    with tf.variable_scope('SEC-PRED-FC-%d' % i):
                        layer_preds_sib = classifiers.hidden(layer_preds_sib, self.hidden_size,
                                                             hidden_func=self.hidden_func,
                                                             hidden_keep_prob=hidden_keep_prob)
                    if self.co_parent:
                        with tf.variable_scope('SEC-CO-ARG-FC-%d' % i):
                            layer_args_cop = classifiers.hidden(layer_args_cop, self.hidden_size,
                                                                hidden_func=self.hidden_func,
                                                                hidden_keep_prob=hidden_keep_prob)
                        with tf.variable_scope('SEC-CO-PRED-FC-%d' % i):
                            layer_preds_cop = classifiers.hidden(layer_preds_cop, self.hidden_size,
                                                                 hidden_func=self.hidden_func,
                                                                 hidden_keep_prob=hidden_keep_prob)
                unary = logits * tf.cast(top_token_weights, dtype=tf.float32)
                with tf.variable_scope('Sibling'):
                    # n x p x k x k
                    layer_sib_score = classifiers.span_trilinear_discriminator(
                        layer_preds_sib, layer_args_sib, layer_args_sib,
                        hidden_keep_prob=hidden_keep_prob_tri,
                        add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
                if self.co_parent:
                    with tf.variable_scope('Co-parent'):
                        # n x p x k x p
                        layer_cop_score = classifiers.span_trilinear_discriminator(
                            layer_preds_cop, layer_args_cop, layer_preds_cop,
                            hidden_keep_prob=hidden_keep_prob_tri,
                            add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
                q_value = unary
                layer_sib_score = layer_sib_score - tf.linalg.band_part(layer_sib_score, -1, 0) + tf.transpose(
                    tf.linalg.band_part(layer_sib_score, 0, -1), perm=[0, 1, 3, 2])
                layer_sib_score = layer_sib_score - tf.linalg.band_part(layer_sib_score, 0, 0)
                layer_sib_mask = tf.expand_dims(top_token_weights, axis=-1) * tf.expand_dims(top_token_weights, axis=-2)
                layer_sib_score = layer_sib_score * tf.to_float(layer_sib_mask)
                if self.co_parent:
                    # n x k x p x p
                    layer_cop_score = tf.transpose(layer_cop_score, perm=[0, 2, 1, 3])
                    layer_cop_score = layer_cop_score - tf.linalg.band_part(layer_cop_score, -1, 0) + tf.transpose(
                        tf.linalg.band_part(layer_cop_score, 0, -1), perm=[0, 1, 3, 2])
                    layer_cop_score = layer_cop_score - tf.linalg.band_part(layer_cop_score, 0, 0)
                    layer_cop_mask = tf.expand_dims(tf.transpose(top_token_weights, perm=[0, 2, 1]),
                                                    axis=-1) * tf.expand_dims(
                        tf.transpose(top_token_weights, perm=[0, 2, 1]), axis=-2)
                    layer_cop_score = layer_cop_score * tf.to_float(layer_cop_mask)
                    # n x p x k x p
                    layer_cop_score = tf.transpose(layer_cop_score, perm=[0, 2, 1, 3])
                for i in range(int(self.num_iteration)):
                    q_value = tf.nn.sigmoid(q_value)
                    outputs['q_value'] = q_value
                    second_temp_sib = tf.einsum('npc,npac->npa', q_value, layer_sib_score)
                    if self.co_parent:
                        # n x p x k
                        second_temp_cop = tf.einsum('nkc,npck->npc', q_value, layer_cop_score)
                    q_value = unary + second_temp_sib
                    if self.co_parent:
                        q_value += second_temp_cop
                logits = q_value

        loss = tf.losses.sigmoid_cross_entropy(top_unlabeled_targets, logits, weights=top_token_weights)

        # -----------------------------------------------------------
        # Process the targets
        # (n x p x k) -> (n x p x k)
        probabilities = tf.nn.sigmoid(logits) * tf.to_float(top_token_weights)
        # (n x p x k) -> (n x p x k)
        predictions = nn.greater(logits, 0, dtype=tf.int32) * top_token_weights  # edge that predicted

        true_positives = predictions * top_unlabeled_targets

        # (n x p x k) -> ()
        n_predictions = tf.reduce_sum(predictions)
        # (n x p x m x m) -> ()
        n_targets = tf.reduce_sum(unlabeled_targets)
        n_true_positives = tf.reduce_sum(true_positives)
        # () - () -> ()
        n_false_positives = n_predictions - n_true_positives
        n_false_negatives = n_targets - n_true_positives
        # (n x p x m x m) -> (n)
        n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])
        # (n x p x k) -> (n)
        n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
        # (n) x 2 -> ()
        n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

        # -----------------------------------------------------------
        # Populate the output dictionary
        outputs['unlabeled_targets'] = unlabeled_targets
        outputs['top_unlabeled_targets'] = top_unlabeled_targets
        outputs['head_probabilities'] = probabilities
        outputs['probabilities'] = probabilities
        outputs['logits'] = logits * tf.to_float(top_token_weights)
        outputs['unlabeled_loss'] = loss
        outputs['loss'] = loss
        outputs['unlabeled_predictions'] = predictions
        outputs['n_unlabeled_true_positives'] = n_true_positives
        outputs['n_unlabeled_false_positives'] = n_false_positives
        outputs['n_unlabeled_false_negatives'] = n_false_negatives
        outputs['n_correct_unlabeled_sequences'] = n_correct_sequences
        outputs['n_top_unlabeled_targets'] = tf.reduce_sum(top_unlabeled_targets)
        outputs['n_unlabeled_predictions'] = tf.reduce_sum(predictions)
        outputs['n_unlabeled_target'] = n_targets
        outputs['n_correct_sequences'] = n_correct_sequences
        outputs['token_weights'] = token_weights
        outputs['top_token_weights'] = top_token_weights
        outputs['top_args_idx'] = top_args_idx
        outputs['candicate_args_mask'] = candicate_args_mask

        if self.second_order:
            outputs['unary'] = unary
            outputs['layer_sib_score'] = layer_sib_score
            outputs['layer_sib_mask'] = layer_sib_mask
            outputs['second_temp_sib'] = second_temp_sib
        if args_embed_type == 'attention':
            outputs['attention_score'] = attention_score
            outputs['mask'] = mask
            outputs['attention_score_new'] = attention_score_new
            outputs['attention_weight'] = attention_weight
            outputs['top_layer_args'] = top_layer_args

        return outputs

    # =============================================================
    def get_bilinear_discriminator_with_args(self, layer, preds, token_weights, top_args_idx, variable_scope=None, reuse=False,
                                    debug=False):
        """"""
        # pdb.set_trace()
        outputs = {}
        hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
        add_linear = self.add_linear


        batch_size, bucket_size, input_size = nn.get_sizes(layer)
        # choose pre-difined args according to top_args_idx
        top_args_nums = nn.get_sizes(top_args_idx)[1]
        # pdb.set_trace()
        one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
        layer_preds = tf.matmul(one_hot_pred, layer)
        pred_nums = nn.get_sizes(layer_preds)[1]

        # layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
        #                         tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)
        # # n x m x m x d -> n x k*d
        # top_layer_args = tf.gather_nd(layer_args, top_args_idx)
        # # n x k x d
        # top_layer_args = tf.reshape(top_layer_args, [batch_size, top_args_nums, input_size*2])

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
            # top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
            # top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])
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
                attention_score_new = mask + attention_score
                attention_weight = tf.math.softmax(attention_score_new, -1)
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

        # elif args_embed_type == ''
        # n x p x m x m
        unlabeled_targets = self.placeholder
        # n x m x m x p
        unlabeled_targets_trans = tf.transpose(unlabeled_targets, [0, 2, 3, 1])
        # n x k*p
        top_unlabeled_targets = tf.gather_nd(unlabeled_targets_trans, top_args_idx)
        # n x p x k
        top_unlabeled_targets = tf.transpose(tf.reshape(top_unlabeled_targets, [batch_size, top_args_nums, pred_nums]),
                                             [0, 2, 1])

        # n x m x m x p
        token_weights_trans = tf.transpose(token_weights, [0, 2, 3, 1])
        # n x k*p
        top_token_weights = tf.gather_nd(token_weights_trans, top_args_idx)
        # n x p x k
        top_token_weights = tf.transpose(tf.reshape(top_token_weights, [batch_size, top_args_nums, pred_nums]),
                                         [0, 2, 1])


        if self.second_order:
            layer_args_sib = top_layer_args  # n x k x d
            layer_preds_sib = layer_preds

            if self.co_parent:
                layer_args_cop = top_layer_args  # n x k x d
                layer_preds_cop = layer_preds

        with tf.variable_scope(variable_scope or self.field):


            for i in six.moves.range(0, self.n_layers):
                with tf.variable_scope('ARGS-FC-%d' % i):  # here is FNN? did not run
                    # n x k x d
                    top_layer_args = classifiers.hidden(top_layer_args, self.hidden_size,
                                                    hidden_func=self.hidden_func,
                                                    hidden_keep_prob=hidden_keep_prob)
                with tf.variable_scope('PRED-FC-%d' % i):
                    # n x p x d
                    layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
                                                     hidden_func=self.hidden_func,
                                                     hidden_keep_prob=hidden_keep_prob)



            with tf.variable_scope('Discriminator'):

                # n x p x k
                logits = classifiers.span_bilinear_discriminator(
                    layer_preds, top_layer_args,
                    hidden_keep_prob=hidden_keep_prob,
                    add_linear=add_linear,
                    target_model = self.target_model,
                    tri_std=self.tri_std_unary)


            if self.second_order:
                hidden_keep_prob_tri = 1 if reuse else self.hidden_keep_prob_tri

                for i in six.moves.range(0, self.n_layers):
                    with tf.variable_scope('SEC-ARG-FC-%d' % i):
                        layer_args_sib = classifiers.hidden(layer_args_sib, self.hidden_size,
                                                            hidden_func=self.hidden_func,
                                                            hidden_keep_prob=hidden_keep_prob)
                    with tf.variable_scope('SEC-PRED-FC-%d' % i):
                        layer_preds_sib = classifiers.hidden(layer_preds_sib, self.hidden_size,
                                                             hidden_func=self.hidden_func,
                                                             hidden_keep_prob=hidden_keep_prob)
                    if self.co_parent:
                        with tf.variable_scope('SEC-CO-ARG-FC-%d' % i):
                            layer_args_cop = classifiers.hidden(layer_args_cop, self.hidden_size,
                                                                hidden_func=self.hidden_func,
                                                                hidden_keep_prob=hidden_keep_prob)
                        with tf.variable_scope('SEC-CO-PRED-FC-%d' % i):
                            layer_preds_cop = classifiers.hidden(layer_preds_cop, self.hidden_size,
                                                                 hidden_func=self.hidden_func,
                                                                 hidden_keep_prob=hidden_keep_prob)
                unary = logits * tf.cast(top_token_weights, dtype=tf.float32)
                with tf.variable_scope('Sibling'):
                    # n x p x k x k
                    layer_sib_score = classifiers.span_trilinear_discriminator(
                        layer_preds_sib, layer_args_sib, layer_args_sib,
                        hidden_keep_prob=hidden_keep_prob_tri,
                        add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
                if self.co_parent:
                    with tf.variable_scope('Co-parent'):
                        # n x p x k x p
                        layer_cop_score = classifiers.span_trilinear_discriminator(
                            layer_preds_cop, layer_args_cop, layer_preds_cop,
                            hidden_keep_prob=hidden_keep_prob_tri,
                            add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
                q_value = unary
                layer_sib_score = layer_sib_score - tf.linalg.band_part(layer_sib_score, -1, 0) + tf.transpose(
                                    tf.linalg.band_part(layer_sib_score, 0, -1), perm=[0, 1, 3, 2])
                layer_sib_score = layer_sib_score - tf.linalg.band_part(layer_sib_score, 0, 0)
                layer_sib_mask = tf.expand_dims(top_token_weights, axis=-1) * tf.expand_dims(top_token_weights, axis=-2)
                layer_sib_score = layer_sib_score * tf.to_float(layer_sib_mask)
                if self.co_parent:
                    # n x k x p x p
                    layer_cop_score = tf.transpose(layer_cop_score,perm=[0,2,1,3])
                    layer_cop_score = layer_cop_score - tf.linalg.band_part(layer_cop_score, -1, 0) + tf.transpose(
                        tf.linalg.band_part(layer_cop_score, 0, -1), perm=[0, 1, 3, 2])
                    layer_cop_score = layer_cop_score - tf.linalg.band_part(layer_cop_score, 0, 0)
                    layer_cop_mask = tf.expand_dims(tf.transpose(top_token_weights,perm=[0,2,1]), axis=-1) * tf.expand_dims(tf.transpose(top_token_weights,perm=[0,2,1]), axis=-2)
                    layer_cop_score = layer_cop_score * tf.to_float(layer_cop_mask)
                    # n x p x k x p
                    layer_cop_score = tf.transpose(layer_cop_score, perm=[0,2,1,3])
                for i in range(int(self.num_iteration)):
                    q_value = tf.nn.sigmoid(q_value)
                    outputs['q_value'] = q_value
                    second_temp_sib = tf.einsum('npc,npac->npa', q_value, layer_sib_score)
                    if self.co_parent:
                        # n x p x k
                        second_temp_cop = tf.einsum('nkc,npck->npc', q_value, layer_cop_score)
                    q_value = unary + second_temp_sib
                    if self.co_parent:
                        q_value += second_temp_cop
                logits = q_value


        loss = tf.losses.sigmoid_cross_entropy(top_unlabeled_targets, logits, weights=top_token_weights)

        # -----------------------------------------------------------
        # Process the targets
        # (n x p x k) -> (n x p x k)
        probabilities = tf.nn.sigmoid(logits) * tf.to_float(top_token_weights)
        # (n x p x k) -> (n x p x k)
        predictions = nn.greater(logits, 0, dtype=tf.int32) * top_token_weights  # edge that predicted

        true_positives = predictions * top_unlabeled_targets

        # (n x p x k) -> ()
        n_predictions = tf.reduce_sum(predictions)
        # (n x p x m x m) -> ()
        n_targets = tf.reduce_sum(unlabeled_targets)
        n_true_positives = tf.reduce_sum(true_positives)
        # () - () -> ()
        n_false_positives = n_predictions - n_true_positives
        n_false_negatives = n_targets - n_true_positives
        # (n x p x m x m) -> (n)
        n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])
        # (n x p x k) -> (n)
        n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
        # (n) x 2 -> ()
        n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

        # -----------------------------------------------------------
        # Populate the output dictionary
        outputs['unlabeled_targets'] = unlabeled_targets
        outputs['top_unlabeled_targets'] = top_unlabeled_targets
        outputs['head_probabilities'] = probabilities
        outputs['probabilities'] = probabilities
        outputs['logits'] = logits * tf.to_float(top_token_weights)
        outputs['unlabeled_loss'] = loss
        outputs['loss'] = loss
        outputs['unlabeled_predictions'] = predictions
        outputs['n_unlabeled_true_positives'] = n_true_positives
        outputs['n_unlabeled_false_positives'] = n_false_positives
        outputs['n_unlabeled_false_negatives'] = n_false_negatives
        outputs['n_correct_unlabeled_sequences'] = n_correct_sequences
        outputs['n_top_unlabeled_targets'] = tf.reduce_sum(top_unlabeled_targets)
        outputs['n_unlabeled_predictions'] = tf.reduce_sum(predictions)
        outputs['n_unlabeled_target'] = n_targets
        outputs['n_correct_sequences'] = n_correct_sequences
        outputs['token_weights'] = token_weights
        outputs['top_token_weights'] = top_token_weights
        outputs['top_args_idx'] = top_args_idx

        if self.second_order:
            outputs['unary'] = unary
            outputs['layer_sib_score'] = layer_sib_score
            outputs['layer_sib_mask'] = layer_sib_mask
            outputs['second_temp_sib'] = second_temp_sib
        if args_embed_type == 'attention':
            outputs['attention_score'] = attention_score
            outputs['mask'] = mask
            outputs['attention_score_new'] = attention_score_new
            outputs['attention_weight'] = attention_weight
            outputs['top_layer_args'] = top_layer_args

        return outputs

    # =============================================================
    def get_bilinear_discriminator_with_args_ppred(self, layer, token_weights, top_args_idx, variable_scope=None,
                                             reuse=False, debug=False):
        """"""
        # pdb.set_trace()
        outputs = {}
        hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
        add_linear = self.add_linear

        batch_size, bucket_size, input_size = nn.get_sizes(layer)
        # choose pre-difined args according to top_args_idx
        top_args_nums = nn.get_sizes(top_args_idx)[1]
        # pdb.set_trace()
        # one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
        layer_preds = layer
        pred_nums = nn.get_sizes(layer_preds)[1]

        # layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
        #                         tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)
        # # n x m x m x d -> n x k*d
        # top_layer_args = tf.gather_nd(layer_args, top_args_idx)
        # # n x k x d
        # top_layer_args = tf.reshape(top_layer_args, [batch_size, top_args_nums, input_size*2])

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
            # top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
            # top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])
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
                attention_score_new = mask + attention_score
                attention_weight = tf.math.softmax(attention_score_new, -1)
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
                                                                                          tf.constant([560, 560, 40, 40]),
                                                                                          -1)
            top_args_end1, top_args_end2, top_args_end3, top_args_end4 = tf.split(top_args_end,
                                                                                  tf.constant([560, 560, 40, 40]), -1)
            top_layer_args = tf.concat([top_args_start1, top_args_end2,
                                        tf.expand_dims(tf.einsum('nkd,nkd->nk', top_args_start3, top_args_end4), -1)], -1)

        # elif args_embed_type == ''
        # n x p x m x m
        unlabeled_targets = self.placeholder
        # n x m x m x p
        unlabeled_targets_trans = tf.transpose(unlabeled_targets, [0, 2, 3, 1])
        # n x k*p
        top_unlabeled_targets = tf.gather_nd(unlabeled_targets_trans, top_args_idx)
        # n x p x k
        top_unlabeled_targets = tf.transpose(tf.reshape(top_unlabeled_targets, [batch_size, top_args_nums, pred_nums]),
                                             [0, 2, 1])

        # n x m x m x p
        token_weights_trans = tf.transpose(token_weights, [0, 2, 3, 1])
        # n x k*p
        top_token_weights = tf.gather_nd(token_weights_trans, top_args_idx)
        # n x p x k
        top_token_weights = tf.transpose(tf.reshape(top_token_weights, [batch_size, top_args_nums, pred_nums]),
                                         [0, 2, 1])

        if self.second_order:
            layer_args_sib = top_layer_args  # n x k x d
            layer_preds_sib = layer_preds

            if self.co_parent:
                layer_args_cop = top_layer_args  # n x k x d
                layer_preds_cop = layer_preds

        with tf.variable_scope(variable_scope or self.field):

            for i in six.moves.range(0, self.n_layers):
                with tf.variable_scope('ARGS-FC-%d' % i):  # here is FNN? did not run
                    # n x k x d
                    top_layer_args = classifiers.hidden(top_layer_args, self.hidden_size,
                                                        hidden_func=self.hidden_func,
                                                        hidden_keep_prob=hidden_keep_prob)
                with tf.variable_scope('PRED-FC-%d' % i):
                    # n x p x d
                    layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
                                                     hidden_func=self.hidden_func,
                                                     hidden_keep_prob=hidden_keep_prob)

            with tf.variable_scope('Discriminator'):

                # n x p x k
                logits = classifiers.span_bilinear_discriminator(
                    layer_preds, top_layer_args,
                    hidden_keep_prob=hidden_keep_prob,
                    add_linear=add_linear,
                    target_model=self.target_model,
                    tri_std=self.tri_std_unary)

            if self.second_order:
                hidden_keep_prob_tri = 1 if reuse else self.hidden_keep_prob_tri

                for i in six.moves.range(0, self.n_layers):
                    with tf.variable_scope('SEC-ARG-FC-%d' % i):
                        layer_args_sib = classifiers.hidden(layer_args_sib, self.hidden_size,
                                                            hidden_func=self.hidden_func,
                                                            hidden_keep_prob=hidden_keep_prob)
                    with tf.variable_scope('SEC-PRED-FC-%d' % i):
                        layer_preds_sib = classifiers.hidden(layer_preds_sib, self.hidden_size,
                                                             hidden_func=self.hidden_func,
                                                             hidden_keep_prob=hidden_keep_prob)
                    if self.co_parent:
                        with tf.variable_scope('SEC-CO-ARG-FC-%d' % i):
                            layer_args_cop = classifiers.hidden(layer_args_cop, self.hidden_size,
                                                                hidden_func=self.hidden_func,
                                                                hidden_keep_prob=hidden_keep_prob)
                        with tf.variable_scope('SEC-CO-PRED-FC-%d' % i):
                            layer_preds_cop = classifiers.hidden(layer_preds_cop, self.hidden_size,
                                                                 hidden_func=self.hidden_func,
                                                                 hidden_keep_prob=hidden_keep_prob)
                unary = logits * tf.cast(top_token_weights, dtype=tf.float32)
                with tf.variable_scope('Sibling'):
                    # n x p x k x k
                    layer_sib_score = classifiers.span_trilinear_discriminator(
                        layer_preds_sib, layer_args_sib, layer_args_sib,
                        hidden_keep_prob=hidden_keep_prob_tri,
                        add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
                if self.co_parent:
                    with tf.variable_scope('Co-parent'):
                        # n x p x k x p
                        layer_cop_score = classifiers.span_trilinear_discriminator(
                            layer_preds_cop, layer_args_cop, layer_preds_cop,
                            hidden_keep_prob=hidden_keep_prob_tri,
                            add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
                q_value = unary
                layer_sib_score = layer_sib_score - tf.linalg.band_part(layer_sib_score, -1, 0) + tf.transpose(
                    tf.linalg.band_part(layer_sib_score, 0, -1), perm=[0, 1, 3, 2])
                layer_sib_score = layer_sib_score - tf.linalg.band_part(layer_sib_score, 0, 0)
                layer_sib_mask = tf.expand_dims(top_token_weights, axis=-1) * tf.expand_dims(top_token_weights, axis=-2)
                layer_sib_score = layer_sib_score * tf.to_float(layer_sib_mask)
                if self.co_parent:
                    # n x k x p x p
                    layer_cop_score = tf.transpose(layer_cop_score, perm=[0, 2, 1, 3])
                    layer_cop_score = layer_cop_score - tf.linalg.band_part(layer_cop_score, -1, 0) + tf.transpose(
                        tf.linalg.band_part(layer_cop_score, 0, -1), perm=[0, 1, 3, 2])
                    layer_cop_score = layer_cop_score - tf.linalg.band_part(layer_cop_score, 0, 0)
                    layer_cop_mask = tf.expand_dims(tf.transpose(top_token_weights, perm=[0, 2, 1]),
                                                    axis=-1) * tf.expand_dims(
                        tf.transpose(top_token_weights, perm=[0, 2, 1]), axis=-2)
                    layer_cop_score = layer_cop_score * tf.to_float(layer_cop_mask)
                    # n x p x k x p
                    layer_cop_score = tf.transpose(layer_cop_score, perm=[0, 2, 1, 3])
                for i in range(int(self.num_iteration)):
                    q_value = tf.nn.sigmoid(q_value)
                    outputs['q_value'] = q_value
                    second_temp_sib = tf.einsum('npc,npac->npa', q_value, layer_sib_score)
                    if self.co_parent:
                        # n x p x k
                        second_temp_cop = tf.einsum('nkc,npck->npc', q_value, layer_cop_score)
                    q_value = unary + second_temp_sib
                    if self.co_parent:
                        q_value += second_temp_cop
                logits = q_value

        loss = tf.losses.sigmoid_cross_entropy(top_unlabeled_targets, logits, weights=top_token_weights)

        # -----------------------------------------------------------
        # Process the targets
        # (n x p x k) -> (n x p x k)
        probabilities = tf.nn.sigmoid(logits) * tf.to_float(top_token_weights)
        # (n x p x k) -> (n x p x k)
        predictions = nn.greater(logits, 0, dtype=tf.int32) * top_token_weights  # edge that predicted

        true_positives = predictions * top_unlabeled_targets

        # (n x p x k) -> ()
        n_predictions = tf.reduce_sum(predictions)
        # (n x p x m x m) -> ()
        n_targets = tf.reduce_sum(unlabeled_targets)
        n_true_positives = tf.reduce_sum(true_positives)
        # () - () -> ()
        n_false_positives = n_predictions - n_true_positives
        n_false_negatives = n_targets - n_true_positives
        # (n x p x m x m) -> (n)
        n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])
        # (n x p x k) -> (n)
        n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
        # (n) x 2 -> ()
        n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

        # -----------------------------------------------------------
        # Populate the output dictionary
        outputs['unlabeled_targets'] = unlabeled_targets
        outputs['top_unlabeled_targets'] = top_unlabeled_targets
        outputs['head_probabilities'] = probabilities
        outputs['probabilities'] = probabilities
        outputs['logits'] = logits * tf.to_float(top_token_weights)
        outputs['unlabeled_loss'] = loss
        outputs['loss'] = loss
        outputs['unlabeled_predictions'] = predictions
        outputs['n_unlabeled_true_positives'] = n_true_positives
        outputs['n_unlabeled_false_positives'] = n_false_positives
        outputs['n_unlabeled_false_negatives'] = n_false_negatives
        outputs['n_correct_unlabeled_sequences'] = n_correct_sequences
        outputs['n_top_unlabeled_targets'] = tf.reduce_sum(top_unlabeled_targets)
        outputs['n_unlabeled_predictions'] = tf.reduce_sum(predictions)
        outputs['n_unlabeled_target'] = n_targets
        outputs['n_correct_sequences'] = n_correct_sequences
        outputs['token_weights'] = token_weights
        outputs['top_token_weights'] = top_token_weights
        outputs['top_args_idx'] = top_args_idx

        if self.second_order:
            outputs['unary'] = unary
            outputs['layer_sib_score'] = layer_sib_score
            outputs['layer_sib_mask'] = layer_sib_mask
            outputs['second_temp_sib'] = second_temp_sib
        if args_embed_type == 'attention':
            outputs['attention_score'] = attention_score
            outputs['mask'] = mask
            outputs['attention_score_new'] = attention_score_new
            outputs['attention_weight'] = attention_weight
            outputs['top_layer_args'] = top_layer_args

        return outputs

    # =============================================================
    def get_bilinear_discriminator_with_args_syntax_ppred(self, layer, syntax_indicator_vocab, syntax_label_vocab, token_weights, top_args_idx, variable_scope=None,
                                                   reuse=False, debug=False):
            """"""
            # pdb.set_trace()
            outputs = {}
            hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
            add_linear = self.add_linear

            batch_size, bucket_size, input_size = nn.get_sizes(layer)
            # choose pre-difined args according to top_args_idx
            top_args_nums = nn.get_sizes(top_args_idx)[1]
            # pdb.set_trace()
            # one_hot_pred = tf.one_hot(preds, depth=bucket_size, dtype=tf.float32)
            layer_preds = layer
            pred_nums = nn.get_sizes(layer_preds)[1]

            # layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
            #                         tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)
            # # n x m x m x d -> n x k*d
            # top_layer_args = tf.gather_nd(layer_args, top_args_idx)
            # # n x k x d
            # top_layer_args = tf.reshape(top_layer_args, [batch_size, top_args_nums, input_size*2])

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


            #--------------------use syntax information, constiuents label----------------------
            syntax_indicator = syntax_indicator_vocab.placeholder
            syntax_indicator = tf.transpose(syntax_indicator,[0,2,1])

            syntax_label = syntax_label_vocab.placeholder
            syntax_label = tf.transpose(syntax_label,[0,2,1])

            # n x k
            top_span_syntax_indicator = tf.gather_nd(syntax_indicator, top_args_idx)
            top_span_syntax_label = tf.gather_nd(syntax_label, top_args_idx)

            # ----------get syntax indicator embedding-------------------------------------
            # embed_keep_prob = 1 if reuse else (embed_keep_prob or self.embed_keep_prob)

            # n x k x 100
            with tf.variable_scope("IndexSyntaxIndicator"):
                top_span_syntax_indicator_embed = embeddings.token_embedding_lookup(2, 50,
                                                          top_span_syntax_indicator,
                                                          nonzero_init=True,
                                                          reuse=reuse)

            with tf.variable_scope("IndexSyntaxLabel"):
                top_span_syntax_label_embed = embeddings.token_embedding_lookup(len(syntax_label_vocab), 50,
                                                                                    top_span_syntax_label,
                                                                                        nonzero_init=True,
                                                                                        reuse=reuse)
                # if embed_keep_prob < 1:
                #     layer = self.drop_func(layer, embed_keep_prob)
            # return layer



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
                # top_args_idx_y = tf.slice(top_args_idx, [0, 0, 1], [-1, -1, 1])
                # top_args_idx_z = tf.slice(top_args_idx, [0, 0, 2], [-1, -1, 1])
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
                    attention_score_new = mask + attention_score
                    attention_weight = tf.math.softmax(attention_score_new, -1)
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
                                                           -1)], -1)

                #-------------------------concat syntax indicator embedding----------------------------------------
                top_layer_args = tf.concat([top_layer_args, top_span_syntax_indicator_embed, top_span_syntax_label_embed], -1)

            # elif args_embed_type == ''
            # n x p x m x m
            unlabeled_targets = self.placeholder
            # n x m x m x p
            unlabeled_targets_trans = tf.transpose(unlabeled_targets, [0, 2, 3, 1])
            # n x k*p
            top_unlabeled_targets = tf.gather_nd(unlabeled_targets_trans, top_args_idx)
            # n x p x k
            top_unlabeled_targets = tf.transpose(
                tf.reshape(top_unlabeled_targets, [batch_size, top_args_nums, pred_nums]),
                [0, 2, 1])

            # n x m x m x p
            token_weights_trans = tf.transpose(token_weights, [0, 2, 3, 1])
            # n x k*p
            top_token_weights = tf.gather_nd(token_weights_trans, top_args_idx)
            # n x p x k
            top_token_weights = tf.transpose(tf.reshape(top_token_weights, [batch_size, top_args_nums, pred_nums]),
                                             [0, 2, 1])

            if self.second_order:
                layer_args_sib = top_layer_args  # n x k x d
                layer_preds_sib = layer_preds

                if self.co_parent:
                    layer_args_cop = top_layer_args  # n x k x d
                    layer_preds_cop = layer_preds

            with tf.variable_scope(variable_scope or self.field):

                for i in six.moves.range(0, self.n_layers):
                    with tf.variable_scope('ARGS-FC-%d' % i):  # here is FNN? did not run
                        # n x k x d
                        top_layer_args = classifiers.hidden(top_layer_args, self.hidden_size,
                                                            hidden_func=self.hidden_func,
                                                            hidden_keep_prob=hidden_keep_prob)
                    with tf.variable_scope('PRED-FC-%d' % i):
                        # n x p x d
                        layer_preds = classifiers.hidden(layer_preds, self.hidden_size,
                                                         hidden_func=self.hidden_func,
                                                         hidden_keep_prob=hidden_keep_prob)

                with tf.variable_scope('Discriminator'):

                    # n x p x k
                    logits = classifiers.span_bilinear_discriminator(
                        layer_preds, top_layer_args,
                        hidden_keep_prob=hidden_keep_prob,
                        add_linear=add_linear,
                        target_model=self.target_model,
                        tri_std=self.tri_std_unary)

                if self.second_order:
                    hidden_keep_prob_tri = 1 if reuse else self.hidden_keep_prob_tri

                    for i in six.moves.range(0, self.n_layers):
                        with tf.variable_scope('SEC-ARG-FC-%d' % i):
                            layer_args_sib = classifiers.hidden(layer_args_sib, self.hidden_size,
                                                                hidden_func=self.hidden_func,
                                                                hidden_keep_prob=hidden_keep_prob)
                        with tf.variable_scope('SEC-PRED-FC-%d' % i):
                            layer_preds_sib = classifiers.hidden(layer_preds_sib, self.hidden_size,
                                                                 hidden_func=self.hidden_func,
                                                                 hidden_keep_prob=hidden_keep_prob)
                        if self.co_parent:
                            with tf.variable_scope('SEC-CO-ARG-FC-%d' % i):
                                layer_args_cop = classifiers.hidden(layer_args_cop, self.hidden_size,
                                                                    hidden_func=self.hidden_func,
                                                                    hidden_keep_prob=hidden_keep_prob)
                            with tf.variable_scope('SEC-CO-PRED-FC-%d' % i):
                                layer_preds_cop = classifiers.hidden(layer_preds_cop, self.hidden_size,
                                                                     hidden_func=self.hidden_func,
                                                                     hidden_keep_prob=hidden_keep_prob)
                    unary = logits * tf.cast(top_token_weights, dtype=tf.float32)
                    with tf.variable_scope('Sibling'):
                        # n x p x k x k
                        layer_sib_score = classifiers.span_trilinear_discriminator(
                            layer_preds_sib, layer_args_sib, layer_args_sib,
                            hidden_keep_prob=hidden_keep_prob_tri,
                            add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
                    if self.co_parent:
                        with tf.variable_scope('Co-parent'):
                            # n x p x k x p
                            layer_cop_score = classifiers.span_trilinear_discriminator(
                                layer_preds_cop, layer_args_cop, layer_preds_cop,
                                hidden_keep_prob=hidden_keep_prob_tri,
                                add_linear=add_linear, tri_std=self.tri_std, hidden_k=self.hidden_k)
                    q_value = unary
                    layer_sib_score = layer_sib_score - tf.linalg.band_part(layer_sib_score, -1, 0) + tf.transpose(
                        tf.linalg.band_part(layer_sib_score, 0, -1), perm=[0, 1, 3, 2])
                    layer_sib_score = layer_sib_score - tf.linalg.band_part(layer_sib_score, 0, 0)
                    layer_sib_mask = tf.expand_dims(top_token_weights, axis=-1) * tf.expand_dims(top_token_weights,
                                                                                                 axis=-2)
                    layer_sib_score = layer_sib_score * tf.to_float(layer_sib_mask)
                    if self.co_parent:
                        # n x k x p x p
                        layer_cop_score = tf.transpose(layer_cop_score, perm=[0, 2, 1, 3])
                        layer_cop_score = layer_cop_score - tf.linalg.band_part(layer_cop_score, -1, 0) + tf.transpose(
                            tf.linalg.band_part(layer_cop_score, 0, -1), perm=[0, 1, 3, 2])
                        layer_cop_score = layer_cop_score - tf.linalg.band_part(layer_cop_score, 0, 0)
                        layer_cop_mask = tf.expand_dims(tf.transpose(top_token_weights, perm=[0, 2, 1]),
                                                        axis=-1) * tf.expand_dims(
                            tf.transpose(top_token_weights, perm=[0, 2, 1]), axis=-2)
                        layer_cop_score = layer_cop_score * tf.to_float(layer_cop_mask)
                        # n x p x k x p
                        layer_cop_score = tf.transpose(layer_cop_score, perm=[0, 2, 1, 3])
                    for i in range(int(self.num_iteration)):
                        q_value = tf.nn.sigmoid(q_value)
                        outputs['q_value'] = q_value
                        second_temp_sib = tf.einsum('npc,npac->npa', q_value, layer_sib_score)
                        if self.co_parent:
                            # n x p x k
                            second_temp_cop = tf.einsum('nkc,npck->npc', q_value, layer_cop_score)
                        q_value = unary + second_temp_sib
                        if self.co_parent:
                            q_value += second_temp_cop
                    logits = q_value

            loss = tf.losses.sigmoid_cross_entropy(top_unlabeled_targets, logits, weights=top_token_weights)

            # -----------------------------------------------------------
            # Process the targets
            # (n x p x k) -> (n x p x k)
            probabilities = tf.nn.sigmoid(logits) * tf.to_float(top_token_weights)
            # (n x p x k) -> (n x p x k)
            predictions = nn.greater(logits, 0, dtype=tf.int32) * top_token_weights  # edge that predicted

            true_positives = predictions * top_unlabeled_targets

            # (n x p x k) -> ()
            n_predictions = tf.reduce_sum(predictions)
            # (n x p x m x m) -> ()
            n_targets = tf.reduce_sum(unlabeled_targets)
            n_true_positives = tf.reduce_sum(true_positives)
            # () - () -> ()
            n_false_positives = n_predictions - n_true_positives
            n_false_negatives = n_targets - n_true_positives
            # (n x p x m x m) -> (n)
            n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1, 2, 3])
            # (n x p x k) -> (n)
            n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1, 2])
            # (n) x 2 -> ()
            n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

            # -----------------------------------------------------------
            # Populate the output dictionary
            outputs['unlabeled_targets'] = unlabeled_targets
            outputs['top_unlabeled_targets'] = top_unlabeled_targets
            outputs['head_probabilities'] = probabilities
            outputs['probabilities'] = probabilities
            outputs['logits'] = logits * tf.to_float(top_token_weights)
            outputs['unlabeled_loss'] = loss
            outputs['loss'] = loss
            outputs['unlabeled_predictions'] = predictions
            outputs['n_unlabeled_true_positives'] = n_true_positives
            outputs['n_unlabeled_false_positives'] = n_false_positives
            outputs['n_unlabeled_false_negatives'] = n_false_negatives
            outputs['n_correct_unlabeled_sequences'] = n_correct_sequences
            outputs['n_top_unlabeled_targets'] = tf.reduce_sum(top_unlabeled_targets)
            outputs['n_unlabeled_predictions'] = tf.reduce_sum(predictions)
            outputs['n_unlabeled_target'] = n_targets
            outputs['n_correct_sequences'] = n_correct_sequences
            outputs['token_weights'] = token_weights
            outputs['top_token_weights'] = top_token_weights
            outputs['top_args_idx'] = top_args_idx
            outputs['top_span_syntax_indicator'] = top_span_syntax_indicator
            outputs['syntax_indicator'] = syntax_indicator

            if self.second_order:
                outputs['unary'] = unary
                outputs['layer_sib_score'] = layer_sib_score
                outputs['layer_sib_mask'] = layer_sib_mask
                outputs['second_temp_sib'] = second_temp_sib
            if args_embed_type == 'attention':
                outputs['attention_score'] = attention_score
                outputs['mask'] = mask
                outputs['attention_score_new'] = attention_score_new
                outputs['attention_weight'] = attention_weight
                outputs['top_layer_args'] = top_layer_args

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
    def target_model(self):
        try:
            return self._config.getstr(self, 'target_model')
        except:
            return ''


    @property
    def tri_std_unary(self):
        try:
            return self._config.getfloat(self, 'tri_std_unary')
        except:
            return 1.0

    @property
    def tri_std(self):
        try:
            return self._config.getfloat(self, 'tri_std')
        except:
            return 0.25

    @property
    def num_iteration(self):
        return self._config.getfloat(self, 'num_iteration')

    @property
    def hidden_k(self):
        try:
            return self._config.getint(self, 'hidden_k')
        except:
            return 200

    @property
    def top_k(self):
        try:
            return self._config.getboolean(self, 'top_k')
        except:
            return False

    @property
    def second_order(self):
        try:
            return self._config.getboolean(self, 'second_order')
        except:
            return False

    @property
    def co_parent(self):
        try:
            return self._config.getboolean(self, 'co_parent')
        except:
            return False

    @property
    def limit_len(self):
        try:
            return self._config.getboolean(self, 'limit_len')
        except:
            return False

    @property
    def top_k_para(self):
        try:
            return self._config.getint(self, 'top_k_para')
        except:
            return 2

    @property
    def hidden_keep_prob_tri(self):
        try:
            return self._config.getfloat(self, 'hidden_keep_prob_tri')
        except:
            return self._config.getfloat(self, 'hidden_keep_prob')

    @property
    def predict_pred(self):
        try:
            return self._config.getboolean(self, 'predict_pred')
        except:
            return False

#***************************************************************
class ArgumentIndexVocab(IndexVocab):

    depth = -5
    # =============================================================
    def __init__(self, *args, **kwargs):
        kwargs['placeholder_shape'] = [None, None, None]
        super(ArgumentIndexVocab, self).__init__(*args, **kwargs)
        self.ROOT_STR = '0'
        self.ROOT_IDX = 0
        return

    # =============================================================
    # token should be: 1:(1,2)-rel|2:(4,6)-acl|5:(8,8)-dep or 1|2|5
    def index(self, token):
        """"""

        nodes = []
        if token != '_':
            token = token.split('|')
            for edge in token:
                head = edge.split(':')[0]
                span_end = edge.split('-', 1)[0].split(':')[1].split(',')[1][:-1]
                nodes.append((int(head), int(span_end)))
        return nodes

    # =============================================================
    # index should be [[1,3], [2,4], [5,3]]
    def token(self, index):
        """"""

        return ['(' + str(head[0]) + ',' + str(head[1]) + ')' for head in index]

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
                    head = edge.split(':')[0]
                    span_end = edge.split('-', 1)[0].split(':')[1].split(',')[1][:-1]
                    nodes.append((int(head), int(span_end)))
            return nodes
        elif hasattr(key, '__iter__'):
            if len(key) > 0:
                if len(key[0]) > 0 and isinstance(key[0][0], six.integer_types + (np.int32, np.int64)):
                    return '|'.join(['(' + str(head[0]) + ',' + str(head[1]) + ')' for head in key])
                else:
                    return [self[k] for k in key]
            else:
                return '_'
        else:
            raise ValueError(
                'Key to SpanIndexVocab.__getitem__ must be (iterable of) strings or iterable of integers list')

    # =============================================================
    def get_discriminator(self, layer, token_weights, variable_scope=None, reuse=False, debug=False):
        outputs = {}

        layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
                                tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)

        hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
        batch_size, bucket_size, _, input_size = nn.get_sizes(layer_args)

        # with tf.variable_scope(variable_scope or self.field):
            # for i in six.moves.range(0, self.n_layers):
            # 	with tf.variable_scope('ARG-FC-%d' % i):  # here is FNN? did not run
            # 		layer = classifiers.hidden(layer, int(self.hidden_size / 2),
            # 		                           hidden_func=self.hidden_func,
            # 		                           hidden_keep_prob=hidden_keep_prob)
            #
            #

        limit_len = 70
        candicate_args_mask = tf.tile(tf.expand_dims(nn.greater(tf.range(bucket_size), 0), 0), [batch_size, 1])
        candicate_args_mask = tf.expand_dims(candicate_args_mask, axis=-1) * tf.expand_dims(candicate_args_mask,
                                                                                            axis=-2)

        candicate_args_mask = tf.cond(bucket_size > tf.constant(limit_len),
                                      lambda: tf.linalg.band_part(candicate_args_mask, 0, limit_len),
                                      lambda: tf.linalg.band_part(candicate_args_mask, 0, -1))
        candicate_args_idx = tf.where(candicate_args_mask)
        # n x num_cand_args x d
        candicate_args_embed = tf.reshape(tf.gather_nd(layer_args, candicate_args_idx),
                                    [batch_size, -1, input_size])
        num_candicate_args = nn.get_sizes(candicate_args_embed)[1]
        index_shape = nn.get_sizes(candicate_args_idx)[-1]
        # n*k x 3
        candicate_args_idx = tf.reshape(candicate_args_idx, [batch_size, num_candicate_args, index_shape])
        token_weights = tf.reshape(tf.gather_nd(token_weights, candicate_args_idx), [batch_size, -1])
        with tf.variable_scope(variable_scope or self.field):
            for i in six.moves.range(0, self.n_layers):
                with tf.variable_scope('ARGS-FC-%d' % i):  # here is FNN? did not run
                    candicate_args_embed = classifiers.hidden(candicate_args_embed, self.hidden_size,
                                                        hidden_keep_prob=hidden_keep_prob)
            # with tf.variable_scope('top'):
            # 	candicate_args_embed = classifiers.hidden(candicate_args_embed, self.hidden_size,
            # 	                                      hidden_func=self.top_hidden_func,
            # 	                                      hidden_keep_prob=hidden_keep_prob)
            with tf.variable_scope('unary_score'):
                unary_args_score = classifiers.hidden(candicate_args_embed, 1,			# b x m x m
                                                      hidden_func=self.top_hidden_func,
                                                      hidden_keep_prob=hidden_keep_prob)

        unary_args_score = tf.reshape(unary_args_score, [batch_size, -1])
        # pdb.set_trace()
        # choose tok_k candicate args
        sel_num = tf.cast(1.*tf.to_float(bucket_size),tf.int32)

        _, idx = tf.math.top_k(unary_args_score, k = sel_num)

        # n x num_cand_args
        candicate_args_mask = tf.reduce_sum(tf.one_hot(idx, num_candicate_args), axis=-2)

        unlabeled_targets = self.placeholder  # n x m x m

        # can not add gold for test
        # gold_args = tf.reshape(tf.gather_nd(unlabeled_targets, candicate_args_idx), [batch_size, -1])
        # candicate_args_mask = tf.to_float(gold_args) + candicate_args_mask
        # candicate_args_mask = tf.tile(tf.expand_dims(tf.reduce_sum(candicate_args_mask,0),0),[batch_size,1])

        top_args_idx = tf.where(candicate_args_mask > 0)


        top_args = tf.gather_nd(candicate_args_idx, top_args_idx)

        # n x k x 3   (1,2,3) sentence(1) span(2,3)
        top_args = tf.reshape(top_args, [batch_size, -1, index_shape])
        # n xk
        top_args_score = tf.reshape(tf.gather_nd(unary_args_score, top_args_idx), [batch_size, -1])




        # unlabeled_targets = self.placeholder  # n x m x m

        # n x num_candicate
        top_unlabeled_targets = tf.reshape(tf.gather_nd(unlabeled_targets, candicate_args_idx),[batch_size, -1])
        # pdb.set_trace()
        loss = tf.losses.sigmoid_cross_entropy(top_unlabeled_targets, unary_args_score, weights=token_weights)

        probabilities = tf.sigmoid(unary_args_score)
        predictions = nn.greater(unary_args_score, 0, dtype=tf.int32) * token_weights  # edge that predicted
        predictions = tf.ones_like(unlabeled_targets)
        predictions = tf.reshape(tf.gather_nd(predictions,top_args),[batch_size,-1])
        top_unlabeled_targets = tf.reshape(tf.gather_nd(unlabeled_targets,top_args),[batch_size,-1])
        true_positives = predictions * top_unlabeled_targets
        n_predictions = tf.reduce_sum(predictions)
        n_targets = tf.reduce_sum(unlabeled_targets)
        n_true_positives = tf.reduce_sum(true_positives)
        # () - () -> ()
        n_false_positives = n_predictions - n_true_positives
        n_false_negatives = n_targets - n_true_positives
        # (n x p x m x m) -> (n)
        n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1,2])
        n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1])
        n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

        outputs['unlabeled_targets'] = unlabeled_targets
        outputs['logits'] = unary_args_score
        outputs['unlabeled_loss'] = loss
        outputs['loss'] = loss
        outputs['probabilities'] = probabilities
        outputs['unlabeled_predictions'] = predictions
        outputs['n_unlabeled_true_positives'] = n_true_positives
        outputs['n_unlabeled_false_positives'] = n_false_positives
        outputs['n_unlabeled_false_negatives'] = n_false_negatives
        outputs['n_correct_unlabeled_sequences'] = n_correct_sequences
        outputs['n_predictions'] = n_predictions
        outputs['n_targets'] = n_targets
        outputs['predictions'] = predictions
        # outputs['predictions_true'] = tf.reduce_sum(predictions_true)
        outputs['n_true_positives'] = n_true_positives
        outputs['n_false_positives'] = n_false_positives
        outputs['n_false_negatives'] = n_false_negatives
        outputs['n_correct_sequences'] = n_correct_sequences
        outputs['token_weights'] = token_weights
        outputs['top_args'] = top_args
        outputs['candicate_args_idx'] = candicate_args_idx
        outputs['top_unlabeled_targets'] = top_unlabeled_targets
        outputs['n_top_unlabeled_targets'] = tf.reduce_sum(top_unlabeled_targets)
        outputs['candicate_args_mask'] = candicate_args_mask

        candicate_args_mask = tf.tile(tf.expand_dims(nn.greater(tf.range(bucket_size), 0), 0), [batch_size, 1])
        candicate_args_mask = tf.expand_dims(candicate_args_mask, axis=-1) * tf.expand_dims(candicate_args_mask,
                                                                                            axis=-2)
        candicate_args_mask = tf.cond(bucket_size > tf.constant(limit_len),
                                      lambda: tf.linalg.band_part(candicate_args_mask, 0, limit_len),
                                      lambda: tf.linalg.band_part(candicate_args_mask, 0, -1))
        outputs['n_candicate_args'] = tf.reduce_sum(candicate_args_mask * unlabeled_targets)
        outputs['num_candicate_args'] = num_candicate_args


        return outputs

    def get_bilinear_discriminator(self, layer, token_weights, variable_scope=None, reuse=False, debug=False):
        outputs = {}
        hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
        n_splits = 2
        with tf.variable_scope(variable_scope or self.field):
            for i in six.moves.range(0, self.n_layers-1):
                with tf.variable_scope('FC-%d' % i):#here is FNN? did not run
                    layer = classifiers.hidden(layer, n_splits*self.hidden_size,
                                                                         hidden_func=self.hidden_func,
                                                                         hidden_keep_prob=hidden_keep_prob)
            with tf.variable_scope('FC-top'):#FNN output and split two layer? FNN+split
                layers = classifiers.hiddens(layer, n_splits*[self.hidden_size],
                                                                         hidden_func=self.hidden_func,
                                                                         hidden_keep_prob=hidden_keep_prob)
            layer1, layer2 = layers.pop(0), layers.pop(0)#layer1 and layer2 are one sentence with different word combination? layer1 head layer2 tail

        with tf.variable_scope('Discriminator'):
            # b x n x n
            unary_args_score = classifiers.bilinear_discriminator(
                layer1, layer2,
                hidden_keep_prob=hidden_keep_prob)

        batch_size, bucket_size, _ = nn.get_sizes(unary_args_score)


        limit_len = 100
        candicate_args_mask = tf.tile(tf.expand_dims(nn.greater(tf.range(bucket_size), 0), 0), [batch_size, 1])
        candicate_args_mask = tf.expand_dims(candicate_args_mask, axis=-1) * tf.expand_dims(candicate_args_mask,
                                                                                            axis=-2)

        candicate_args_mask = tf.cond(bucket_size > tf.constant(limit_len),
                                      lambda: tf.linalg.band_part(candicate_args_mask, 0, limit_len),
                                      lambda: tf.linalg.band_part(candicate_args_mask, 0, -1))


        unlabeled_targets = self.placeholder
        loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets, unary_args_score, weights=token_weights)

        # token_weights = token_weights * candicate_args_mask
        unary_args_score_mask = tf.cast((1 - candicate_args_mask*token_weights), tf.float32) * (-1e13)
        unary_args_score_trans = unary_args_score + unary_args_score_mask

        unary_args_score_trans = tf.reshape(unary_args_score_trans,[batch_size,-1])
        sel_num = tf.cast(1. * tf.to_float(bucket_size), tf.int32)
        _, ids = tf.math.top_k(unary_args_score_trans, sel_num)
        ids_x = tf.cast(ids/bucket_size, tf.int32)
        ids_y = ids%bucket_size
        ids_sent = tf.tile(tf.expand_dims(tf.range(batch_size),-1),[1,sel_num])
        ids_x = tf.reshape(ids_x,[-1,1])
        ids_y = tf.reshape(ids_y, [-1, 1])
        ids_sent = tf.reshape(ids_sent, [-1, 1])
        top_args = tf.concat([ids_sent,ids_x,ids_y],axis=-1)
        top_args = tf.reshape(top_args,[batch_size,sel_num,3])





        #
        # layer_args = tf.concat([tf.expand_dims(layer, axis=-2) - tf.expand_dims(layer, axis=-3),
        #                         tf.expand_dims(layer, axis=-2) + tf.expand_dims(layer, axis=-3)], axis=-1)
        #
        #
        # batch_size, bucket_size, _, input_size = nn.get_sizes(layer_args)
        #
        # # with tf.variable_scope(variable_scope or self.field):
        #     # for i in six.moves.range(0, self.n_layers):
        #     # 	with tf.variable_scope('ARG-FC-%d' % i):  # here is FNN? did not run
        #     # 		layer = classifiers.hidden(layer, int(self.hidden_size / 2),
        #     # 		                           hidden_func=self.hidden_func,
        #     # 		                           hidden_keep_prob=hidden_keep_prob)
        #     #
        #     #
        #
        # limit_len = 70
        # candicate_args_mask = tf.tile(tf.expand_dims(nn.greater(tf.range(bucket_size), 0), 0), [batch_size, 1])
        # candicate_args_mask = tf.expand_dims(candicate_args_mask, axis=-1) * tf.expand_dims(candicate_args_mask,
        #                                                                                     axis=-2)
        #
        # candicate_args_mask = tf.cond(bucket_size > tf.constant(limit_len),
        #                               lambda: tf.linalg.band_part(candicate_args_mask, 0, limit_len),
        #                               lambda: tf.linalg.band_part(candicate_args_mask, 0, -1))
        # candicate_args_idx = tf.where(candicate_args_mask)
        # # n x num_cand_args x d
        # candicate_args_embed = tf.reshape(tf.gather_nd(layer_args, candicate_args_idx),
        #                             [batch_size, -1, input_size])
        # num_candicate_args = nn.get_sizes(candicate_args_embed)[1]
        # index_shape = nn.get_sizes(candicate_args_idx)[-1]
        # # n*k x 3
        # candicate_args_idx = tf.reshape(candicate_args_idx, [batch_size, num_candicate_args, index_shape])
        # token_weights = tf.reshape(tf.gather_nd(token_weights, candicate_args_idx), [batch_size, -1])
        # with tf.variable_scope(variable_scope or self.field):
        #     for i in six.moves.range(0, self.n_layers):
        #         with tf.variable_scope('ARGS-FC-%d' % i):  # here is FNN? did not run
        #             candicate_args_embed = classifiers.hidden(candicate_args_embed, self.hidden_size,
        #                                                 hidden_keep_prob=hidden_keep_prob)
        #     # with tf.variable_scope('top'):
        #     # 	candicate_args_embed = classifiers.hidden(candicate_args_embed, self.hidden_size,
        #     # 	                                      hidden_func=self.top_hidden_func,
        #     # 	                                      hidden_keep_prob=hidden_keep_prob)
        #     with tf.variable_scope('unary_score'):
        #         unary_args_score = classifiers.hidden(candicate_args_embed, 1,			# b x m x m
        #                                               hidden_func=self.top_hidden_func,
        #                                               hidden_keep_prob=hidden_keep_prob)
        #
        # unary_args_score = tf.reshape(unary_args_score, [batch_size, -1])
        # # pdb.set_trace()
        # # choose tok_k candicate args
        # sel_num = tf.cast(1.*tf.to_float(bucket_size),tf.int32)
        #
        # _, idx = tf.math.top_k(unary_args_score, k = sel_num)
        #
        # # n x num_cand_args
        # candicate_args_mask = tf.reduce_sum(tf.one_hot(idx, num_candicate_args), axis=-2)
        #
        # unlabeled_targets = self.placeholder  # n x m x m
        #
        # # can not add gold for test
        # # gold_args = tf.reshape(tf.gather_nd(unlabeled_targets, candicate_args_idx), [batch_size, -1])
        # # candicate_args_mask = tf.to_float(gold_args) + candicate_args_mask
        # # candicate_args_mask = tf.tile(tf.expand_dims(tf.reduce_sum(candicate_args_mask,0),0),[batch_size,1])
        #
        # top_args_idx = tf.where(candicate_args_mask > 0)
        #
        #
        # top_args = tf.gather_nd(candicate_args_idx, top_args_idx)
        #
        # # n x k x 3   (1,2,3) sentence(1) span(2,3)
        # top_args = tf.reshape(top_args, [batch_size, -1, index_shape])
        # # n xk
        # top_args_score = tf.reshape(tf.gather_nd(unary_args_score, top_args_idx), [batch_size, -1])
        #
        #
        #
        #
        # # unlabeled_targets = self.placeholder  # n x m x m
        #
        # # n x num_candicate
        # top_unlabeled_targets = tf.reshape(tf.gather_nd(unlabeled_targets, candicate_args_idx),[batch_size, -1])
        # # pdb.set_trace()
        # loss = tf.losses.sigmoid_cross_entropy(top_unlabeled_targets, unary_args_score, weights=token_weights)

        probabilities = tf.sigmoid(unary_args_score)
        # predictions = nn.greater(unary_args_score, 0, dtype=tf.int32) * token_weights  # edge that predicted
        predictions = tf.ones_like(unlabeled_targets)
        predictions = tf.reshape(tf.gather_nd(predictions,top_args),[batch_size,-1])
        top_unlabeled_targets = tf.reshape(tf.gather_nd(unlabeled_targets,top_args),[batch_size,-1])
        true_positives = predictions * top_unlabeled_targets
        n_predictions = tf.reduce_sum(predictions)
        n_targets = tf.reduce_sum(unlabeled_targets)
        n_true_positives = tf.reduce_sum(true_positives)
        # () - () -> ()
        n_false_positives = n_predictions - n_true_positives
        n_false_negatives = n_targets - n_true_positives
        # (n x p x m x m) -> (n)
        n_targets_per_sequence = tf.reduce_sum(unlabeled_targets, axis=[1,2])
        n_true_positives_per_sequence = tf.reduce_sum(true_positives, axis=[1])
        n_correct_sequences = tf.reduce_sum(nn.equal(n_true_positives_per_sequence, n_targets_per_sequence))

        outputs['unlabeled_targets'] = unlabeled_targets
        outputs['logits'] = unary_args_score
        outputs['unlabeled_loss'] = loss
        outputs['loss'] = loss
        outputs['probabilities'] = probabilities
        outputs['unlabeled_predictions'] = predictions
        outputs['n_unlabeled_true_positives'] = n_true_positives
        outputs['n_unlabeled_false_positives'] = n_false_positives
        outputs['n_unlabeled_false_negatives'] = n_false_negatives
        outputs['n_correct_unlabeled_sequences'] = n_correct_sequences
        outputs['n_predictions'] = n_predictions
        outputs['n_targets'] = n_targets
        outputs['predictions'] = predictions
        # outputs['predictions_true'] = tf.reduce_sum(predictions_true)
        outputs['n_true_positives'] = n_true_positives
        outputs['n_false_positives'] = n_false_positives
        outputs['n_false_negatives'] = n_false_negatives
        outputs['n_correct_sequences'] = n_correct_sequences
        outputs['token_weights'] = token_weights
        outputs['top_args'] = top_args

        outputs['top_unlabeled_targets'] = top_unlabeled_targets
        outputs['n_top_unlabeled_targets'] = tf.reduce_sum(top_unlabeled_targets)
        outputs['candicate_args_mask'] = candicate_args_mask

        return outputs

    @property
    def top_hidden_func(self):
        hidden_func = self._config.getstr(self, 'top_hidden_func')
        if hasattr(nonlin, hidden_func):
            return getattr(nonlin, hidden_func)
        else:
            raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, hidden_func))
#***************************************************************

class GraphIndexVocab(IndexVocab):
    """"""

    _depth = -1

    #=============================================================
    def __init__(self, *args, **kwargs):
        """"""

        kwargs['placeholder_shape'] = [None, None, None]
        super(GraphIndexVocab, self).__init__(*args, **kwargs)
        self.ROOT_STR = '0'
        self.ROOT_IDX = 0
        return

    #=============================================================
    def get_bilinear_discriminator(self, layer, token_weights, variable_scope=None, reuse=False, debug=False):
        """"""
        #pdb.set_trace()
        outputs = {}
        recur_layer = layer
        hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
        add_linear = self.add_linear
        n_splits = 2*(1+self.linearize+self.distance)
        with tf.variable_scope(variable_scope or self.field):
            for i in six.moves.range(0, self.n_layers-1):
                with tf.variable_scope('FC-%d' % i):#here is FNN? did not run
                    layer = classifiers.hidden(layer, n_splits*self.hidden_size,
                                                                         hidden_func=self.hidden_func,
                                                                         hidden_keep_prob=hidden_keep_prob)
            with tf.variable_scope('FC-top'):#FNN output and split two layer? FNN+split
                layers = classifiers.hiddens(layer, n_splits*[self.hidden_size],
                                                                         hidden_func=self.hidden_func,
                                                                         hidden_keep_prob=hidden_keep_prob)
            layer1, layer2 = layers.pop(0), layers.pop(0)#layer1 and layer2 are one sentence with different word combination? layer1 head layer2 tail
            if self.linearize:#false
                lin_layer1, lin_layer2 = layers.pop(0), layers.pop(0)
            if self.distance:#false in graph
                dist_layer1, dist_layer2 = layers.pop(0), layers.pop(0)

            with tf.variable_scope('Discriminator'):
                if self.diagonal:
                    logits = classifiers.diagonal_bilinear_discriminator(
                        layer1, layer2,
                        hidden_keep_prob=hidden_keep_prob,
                        add_linear=add_linear)
                    if self.linearize:
                        with tf.variable_scope('Linearization'):
                            lin_logits = classifiers.diagonal_bilinear_discriminator(
                                lin_layer1, lin_layer2,
                                hidden_keep_prob=hidden_keep_prob,
                                add_linear=add_linear)
                    if self.distance:
                        with tf.variable_scope('Distance'):
                            dist_lamda = 1+tf.nn.softplus(classifiers.diagonal_bilinear_discriminator(
                                dist_layer1, dist_layer2,
                                hidden_keep_prob=hidden_keep_prob,
                                add_linear=add_linear))
                else:
                    #only run here
                    logits = classifiers.bilinear_discriminator(
                        layer1, layer2,
                        hidden_keep_prob=hidden_keep_prob,
                        add_linear=add_linear)
                    if self.linearize:
                        with tf.variable_scope('Linearization'):
                            lin_logits = classifiers.bilinear_discriminator(
                                lin_layer1, lin_layer2,
                                hidden_keep_prob=hidden_keep_prob,
                                add_linear=add_linear)
                    if self.distance:
                        with tf.variable_scope('Distance'):
                            dist_lamda = 1+tf.nn.softplus(classifiers.bilinear_discriminator(
                                dist_layer1, dist_layer2,
                                hidden_keep_prob=hidden_keep_prob,
                                add_linear=add_linear))

                #-----------------------------------------------------------
                # Process the targets
                # (n x m x m) -> (n x m x m)
                #here in fact is a graph, which is m*m representing the connection between each edge
                unlabeled_targets = self.placeholder#ground truth graph, what is self.placeholder?
                #USELESS
                shape = tf.shape(layer1)
                batch_size, bucket_size = shape[0], shape[1]
                # (1 x m)
                ids = tf.expand_dims(tf.range(bucket_size), 0)
                # (1 x m) -> (1 x 1 x m)
                head_ids = tf.expand_dims(ids, -2)
                # (1 x m) -> (1 x m x 1)
                dep_ids = tf.expand_dims(ids, -1)


                #no running here
                if self.linearize:#So what is linearize? The linear part of bilinear?
                    # Wherever the head is to the left
                    # (n x m x m), (1 x m x 1) -> (n x m x m)
                    lin_targets = tf.to_float(tf.less(unlabeled_targets, dep_ids))
                    # cross-entropy of the linearization of each i,j pair
                    # (1 x 1 x m), (1 x m x 1) -> (n x m x m)
                    lin_ids = tf.tile(tf.less(head_ids, dep_ids), [batch_size, 1, 1])
                    # (n x 1 x m), (n x m x 1) -> (n x m x m)
                    lin_xent = -tf.nn.softplus(tf.where(lin_ids, -lin_logits, lin_logits))
                    # add the cross-entropy to the logits
                    # (n x m x m), (n x m x m) -> (n x m x m)
                    logits += tf.stop_gradient(lin_xent)
                if self.distance:
                    # (n x m x m) - (1 x m x 1) -> (n x m x m)
                    dist_targets = tf.abs(unlabeled_targets - dep_ids)
                    # KL-divergence of the distance of each i,j pair
                    # (1 x 1 x m) - (1 x m x 1) -> (n x m x m)
                    dist_ids = tf.to_float(tf.tile(tf.abs(head_ids - dep_ids), [batch_size, 1, 1]))+1e-12
                    # (n x m x m), (n x m x m) -> (n x m x m)
                    #dist_kld = (dist_ids * tf.log(dist_lamda / dist_ids) + dist_ids - dist_lamda)
                    dist_kld = -tf.log((dist_ids - dist_lamda)**2/2 + 1)
                    # add the KL-divergence to the logits
                    # (n x m x m), (n x m x m) -> (n x m x m)
                    logits += tf.stop_gradient(dist_kld)


                if debug:
                    outputs['printdata']={}
                    outputs['printdata']['logits']=logits
                #-----------------------------------------------------------
                # Compute probabilities/cross entropy
                # (n x m x m) -> (n x m x m)
                probabilities = tf.nn.sigmoid(logits) * tf.to_float(token_weights)#token weights is sentence length?
                # (n x m x m), (n x m x m), (n x m x m) -> ()
                # loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets, logits, weights=token_weights)#here label_smoothing is 0, the sigmoid XE have any effect?

                #======================================================
                # try balance cross entropy
                if self.balance:
                    # pdb.set_trace()
                    balance_weights = tf.where(tf.equal(unlabeled_targets,1),tf.to_float(tf.ones_like(unlabeled_targets))*tf.constant(2*self.alpha),tf.to_float(tf.ones_like(unlabeled_targets))*tf.constant(2*(1-self.alpha)))
                    loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets, logits, weights=tf.to_float(token_weights)*balance_weights)#here label_smoothing is 0, the sigmoid XE have any effect?
                elif self.dl_loss:
                    gamma = self.gamma
                    loss = nn.dsc_loss(unlabeled_targets, probabilities, tf.to_float(token_weights), gamma)
                else:
                    loss = tf.losses.sigmoid_cross_entropy(unlabeled_targets, logits, weights=token_weights)#here label_smoothing is 0, the sigmoid XE have any effect?

                #=======================================================

                n_tokens = tf.to_float(tf.reduce_sum(token_weights))
                if self.linearize:
                    lin_target_xent = lin_xent * unlabeled_targets
                    loss -= tf.reduce_sum(lin_target_xent * tf.to_float(token_weights)) / (n_tokens + 1e-12)
                if self.distance:
                    dist_target_kld = dist_kld * unlabeled_targets
                    loss -= tf.reduce_sum(dist_target_kld * tf.to_float(token_weights)) / (n_tokens + 1e-12)

                #-----------------------------------------------------------
                # Compute predictions/accuracy
                # precision/recall
                # (n x m x m) -> (n x m x m)
                predictions = nn.greater(logits, 0, dtype=tf.int32) * token_weights#edge that predicted
                # if self.compare_precision:
                # 		#pdb.set_trace()
                # 		# (n x m x m) -> (n x m)
                # 		temp_predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # 		# (n x m) -> (n x m x m)
                # 		cond = tf.equal(logits, tf.expand_dims(tf.reduce_max(logits,-1),-1))
                # 		predictions = tf.where(cond, tf.cast(cond,tf.float32), tf.zeros_like(logits))
                # 		predictions = tf.cast(predictions,tf.int32) * token_weights
                # 		# # (n x m) (*) (n x m) -> (n x m)
                # 		# n_true_positives = tf.reduce_sum(nn.equal(tf.argmax(unlabeled_targets,axis=-1, output_type=tf.int32), temp_predictions) * self.token_weights)
                # 		# n_predictions_temp = tf.reduce_sum(temp_predictions)
                # 		# n_false_positives = n_predictions_temp - n_true_positives

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
        outputs['unlabeled_targets'] = unlabeled_targets
        outputs['head_probabilities'] = probabilities
        outputs['probabilities'] = probabilities
        outputs['logits'] = logits*tf.to_float(token_weights)
        outputs['unlabeled_loss'] = loss
        outputs['loss'] = loss
        if debug:
            outputs['temp_targets'] = tf.argmax(unlabeled_targets,axis=-1, output_type=tf.int32)
            # outputs['temp_predictions'] = temp_predictions
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
    # token should be: 1:rel|2:acl|5:dep or 1|2|5
    def index(self, token):
        """"""

        nodes = []
        if token != '_':
            token = token.split('|')
            for edge in token:
                head = edge.split(':')[0]
                nodes.append(int(head))
        return nodes

    #=============================================================
    # index should be [1, 2, 5]
    def token(self, index):
        """"""

        return [str(head) for head in index]

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
                    head = edge.split(':')[0]
                    nodes.append(int(head))
            return nodes
        elif hasattr(key, '__iter__'):
            if len(key) > 0:
                if isinstance(key[0], six.integer_types + (np.int32, np.int64)):
                    return '|'.join([str(head) for head in key])
                else:
                    return [self[k] for k in key]
            else:
                return '_'
        else:
            raise ValueError('Key to GraphIndexVocab.__getitem__ must be (iterable of) strings or iterable of integers')

    @property
    def alpha(self):
        return self._config.getfloat(self, 'alpha')
    @property
    def balance(self):
        try:
            return self._config.getboolean(self, 'balance')
        except:
            return False

    @property
    def dl_loss(self):
        try:
            return self._config.getboolean(self, 'dl_loss')
        except:
            return False

    @property
    def gamma(self):
        return self._config.getfloat(self, 'gamma')


#***************************************************************
class IDIndexVocab(IndexVocab, cv.IDVocab):
    pass
class DepheadIndexVocab(IndexVocab, cv.DepheadVocab):
    pass
class SemheadGraphIndexVocab(GraphIndexVocab, cv.SemheadVocab):
    pass
class SpanheadGraphIndexVocab(SpanIndexVocab, cv.SpanheadVocab):
    pass
class ArgumentGraphIndexVocab(ArgumentIndexVocab, cv.ArgumentVocab):
    pass
class SpanendGraphIndexVocab(GraphIndexVocab, cv.SpanendVocab):
    pass
class PredIndexVocab(PredPadVocab, cv.PredVocab):
    pass
