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

import numpy as np
import tensorflow as tf

from parser.base_network import BaseNetwork
from parser.neural import nn, nonlin, embeddings, recurrent, classifiers
from parser.structs.vocabs.pointer_generator import PointerGenerator
import pdb


# ***************************************************************
class UnfactorGraphParserNetwork(BaseNetwork):
    """"""

    # =============================================================
    def build_graph(self, input_network_outputs={}, reuse=True, debug=False, nornn=False):
        """"""
        # pdb.set_trace()

        with tf.variable_scope('Embeddings'):

            if self.sum_pos:  # TODO this should be done with a `POSMultivocab`
                pos_vocabs = list(filter(lambda x: 'POS' in x.classname, self.input_vocabs))
                pos_tensors = [input_vocab.get_input_tensor(embed_keep_prob=1, reuse=reuse) for input_vocab in
                               pos_vocabs]
                non_pos_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs if
                                   'POS' not in input_vocab.classname]
                # pos_tensors = [tf.Print(pos_tensor, [pos_tensor]) for pos_tensor in pos_tensors]
                # non_pos_tensors = [tf.Print(non_pos_tensor, [non_pos_tensor]) for non_pos_tensor in non_pos_tensors]
                if pos_tensors:
                    pos_tensors = tf.add_n(pos_tensors)
                    if not reuse:
                        pos_tensors = [pos_vocabs[0].drop_func(pos_tensors, pos_vocabs[0].embed_keep_prob)]
                    else:
                        pos_tensors = [pos_tensors]
                input_tensors = non_pos_tensors + pos_tensors
            elif self.no_flag:
                input_tensors = []
                for input_vocab in self.input_vocabs:
                    # pdb.set_trace()
                    if input_vocab.field == 'flag' or input_vocab.field == 'predicate':
                        continue
                    input_tensors.append(input_vocab.get_input_tensor(reuse=reuse))
                # pdb.set_trace()
            else:  # run this
                input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs]
            # pdb.set_trace()
            for input_network, output in input_network_outputs:
                with tf.variable_scope(input_network.classname):
                    input_tensors.append(input_network.get_input_tensor(output, reuse=reuse))
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
        # ==========================================================================================
        if self.mask:
            for input_vocab in self.input_vocabs:
                # pdb.set_trace()
                if input_vocab.classname == 'FlagTokenVocab':
                    predicate_weights = nn.greater(input_vocab.placeholder, 5)

                    predicate_weights_ex = tf.tile(predicate_weights, [1, bucket_size])
                    predicate_weights_ex = nn.reshape(predicate_weights_ex, [batch_size, bucket_size, bucket_size])
                    zero = tf.zeros([batch_size, bucket_size, bucket_size - 1], dtype=tf.int32)
                    root_p = tf.expand_dims(predicate_weights, axis=1)
                    root_p = tf.transpose(root_p, [0, 2, 1])
                    root_p = tf.concat([root_p, zero], 2)
                    token_weights3D = token_weights3D * (predicate_weights_ex + root_p)
                    break
        if self.mask_decoder and reuse:
            for input_vocab in self.input_vocabs:
                # pdb.set_trace()
                if input_vocab.classname == 'FlagTokenVocab':
                    predicate_weights = nn.greater(input_vocab.placeholder, 5)
                    predicate_weights_ex = tf.tile(predicate_weights, [1, bucket_size])
                    predicate_weights_ex = nn.reshape(predicate_weights_ex, [batch_size, bucket_size, bucket_size])
                    zero = tf.zeros([batch_size, bucket_size, bucket_size - 1], dtype=tf.int32)
                    root_p = tf.expand_dims(predicate_weights, axis=1)
                    root_p = tf.transpose(root_p, [0, 2, 1])
                    root_p = tf.concat([root_p, zero], 2)
                    token_weights_decoder = token_weights3D * (predicate_weights_ex + root_p)
                    break

        # ===========================================================================================

        if self.mask_decoder and reuse:
            tokens = {'n_tokens': n_tokens,
                      'tokens_per_sequence': tokens_per_sequence,
                      'input_tensors': input_tensors,
                      'token_weights': token_weights,
                      'token_weights3D': token_weights3D,
                      'token_weights_decoder': token_weights_decoder,
                      'n_sequences': n_sequences}
            # 'tt': predicate_weights_ex + root_p}
        else:
            tokens = {'n_tokens': n_tokens,
                      'tokens_per_sequence': tokens_per_sequence,
                      'input_tensors': input_tensors,
                      'token_weights': token_weights,
                      'token_weights3D': token_weights3D,
                      'n_sequences': n_sequences}

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

        refine = self.refine
        # layers
        with tf.variable_scope('Classifiers'):
            if 'semrel' in output_fields:
                vocab = output_fields['semrel']
                head_vocab = output_fields['semhead']
                unlabeled_outputs = None
                if vocab.factorized:
                    with tf.variable_scope('Unlabeled'):
                        # pdb.set_trace()
                        if self.layer_mask(head_vocab):
                            unlabeled_outputs = head_vocab.get_bilinear_discriminator(
                                layer,
                                token_weights=token_weights3D,
                                reuse=reuse, debug=debug, token_weights4D=token_weights4D)
                        # ------------ SRL mask decoder ---------------------------------------
                        elif self.mask_decoder and reuse:
                            unlabeled_outputs = head_vocab.get_bilinear_discriminator(
                                layer,
                                token_weights=token_weights3D,
                                reuse=reuse, debug=debug)
                        # ------------ SRL mask decoder ---------------------------------------
                        else:
                            unlabeled_outputs = head_vocab.get_bilinear_discriminator(
                                layer,
                                token_weights=token_weights3D,
                                reuse=reuse, debug=debug)

                    with tf.variable_scope('Labeled'):
                        labeled_outputs = vocab.get_unfactored_bilinear_classifier_test(layer, unlabeled_outputs,refine,
                                                                                   token_weights=token_weights3D,
                                                                                   reuse=reuse)
                else:

                    # labeled_outputs = vocab.get_unfactored_bilinear_classifier(layer,
                    #                                                            token_weights=token_weights3D,
                    #                                                            reuse=reuse)
                    with tf.variable_scope('BalanceLabeled'):
                        balance_labeled_outputs = vocab.get_unfactored_bilinear_classifier_balance(layer, token_weights=token_weights3D,reuse=reuse)
                    

                    with tf.variable_scope('Labeled'):
                        labeled_outputs = vocab.get_unfactored_bilinear_classifier_test(layer, balance_labeled_outputs,
                                                                                        refine,
                                                                                        token_weights=token_weights3D,
                                                                                        reuse=reuse)

                outputs['semgraph'] = labeled_outputs

                self._evals.add('semgraph')
            elif 'semhead' in output_fields:
                vocab = output_fields['semhead']
                outputs[vocab.classname] = vocab.get_bilinear_classifier(
                    layer,
                    token_weights=token_weights3D,
                    reuse=reuse)
                self._evals.add('semhead')



        return outputs, tokens

    # =============================================================
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
    def refine(self):
        try:
            return self._config.getboolean(self, 'refine')
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
