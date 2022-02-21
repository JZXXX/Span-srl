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
from parser.structs.vocabs.token_vocabs import TokenVocab
import pdb



# ***************************************************************
class TriGraphParserNetwork(BaseNetwork):
    """"""

    # =============================================================
    def build_graph(self, input_network_outputs={}, reuse=True, debug=False, nornn=False):
        """"""
        # pdb.set_trace()
        # with tf.variable_scope('Embeddings'):
        #
        #     input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs]
        #     # pdb.set_trace()
        #     layer = tf.concat(input_tensors, 2)  # batch*sentence*feature? or batch* sentence^2*feature?
        # # pdb.set_trace()
        # n_nonzero = tf.to_float(tf.count_nonzero(layer, axis=-1, keepdims=True))
        # batch_size, bucket_size, input_size = nn.get_sizes(layer)
        # layer *= input_size / (n_nonzero + tf.constant(1e-12))
        #
        # token_weights = nn.greater(self.id_vocab.placeholder, 0)  # find sentence length
        # tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
        # n_tokens = tf.reduce_sum(tokens_per_sequence)
        # n_sequences = tf.count_nonzero(tokens_per_sequence)
        # output_fields = {vocab.field: vocab for vocab in self.output_vocabs}
        # outputs = {}
        # # pdb.set_trace()
        # root_weights = token_weights + (1 - nn.greater(tf.range(bucket_size), 0))
        # token_weights3D = tf.expand_dims(token_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
        #
        # # ==========================================================================================
        # if self.mask:
        #     for input_vocab in self.input_vocabs:
        #         # pdb.set_trace()
        #         if input_vocab.classname == 'FlagTokenVocab':
        #             predicate_weights = nn.greater(input_vocab.placeholder, 5)
        #
        #             predicate_weights_ex = tf.tile(predicate_weights, [1, bucket_size])
        #             predicate_weights_ex = nn.reshape(predicate_weights_ex, [batch_size, bucket_size, bucket_size])
        #             zero = tf.zeros([batch_size, bucket_size, bucket_size - 1], dtype=tf.int32)
        #             root_p = tf.expand_dims(predicate_weights, axis=1)
        #             root_p = tf.transpose(root_p, [0, 2, 1])
        #             root_p = tf.concat([root_p, zero], 2)
        #             token_weights3D = token_weights3D * (predicate_weights_ex + root_p)
        #             break
        # if self.mask_decoder and reuse:
        #     for input_vocab in self.input_vocabs:
        #         # pdb.set_trace()
        #         if input_vocab.classname == 'FlagTokenVocab':
        #             predicate_weights = nn.greater(input_vocab.placeholder, 5)
        #             predicate_weights_ex = tf.tile(predicate_weights, [1, bucket_size])
        #             predicate_weights_ex = nn.reshape(predicate_weights_ex, [batch_size, bucket_size, bucket_size])
        #             zero = tf.zeros([batch_size, bucket_size, bucket_size - 1], dtype=tf.int32)
        #             root_p = tf.expand_dims(predicate_weights, axis=1)
        #             root_p = tf.transpose(root_p, [0, 2, 1])
        #             root_p = tf.concat([root_p, zero], 2)
        #             token_weights3D = token_weights3D * (predicate_weights_ex + root_p)
        #             break
        #
        # # ===========================================================================================
        #

        # token_weights2D = tf.expand_dims(root_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
        # # as our three dimension a b c, is a->b to deciding, so all binary potential should not contain root(x)
        # # in fact root should contained in second order prediction except sibling, but for simpler we set all for same
        # token_weights4D = tf.cast(
        #     tf.expand_dims(token_weights2D, axis=-3) * tf.expand_dims(tf.expand_dims(root_weights, axis=-1), axis=-1),
        #     dtype=tf.float32)
        # # abc -> ab,ac
        # # token_weights_sib = tf.cast(tf.expand_dims(root_, axis=-3) * tf.expand_dims(tf.expand_dims(root_weights, axis=-1),axis=-1),dtype=tf.float32)
        # # abc -> ab,cb
        # # pdb.set_trace()
        # token_weights_cop = tf.cast(
        #     tf.expand_dims(token_weights2D, axis=-2) * tf.expand_dims(tf.expand_dims(token_weights, axis=1), axis=-1),
        #     dtype=tf.float32)
        # token_weights_cop_0 = token_weights_cop[:, 0] * tf.cast(tf.transpose(token_weights3D, [0, 2, 1]),
        #                                                         dtype=tf.float32)
        # token_weights_cop = tf.concat([token_weights_cop_0[:, None, :], token_weights_cop[:, 1:]], 1)
        # # data=np.stack((devprint['printdata']['layer_cop'][0][0]*devprint['token_weights3D'][0].T)[None,:],devprint['printdata']['layer_cop'][0][1:])
        # # abc -> ab, bc
        # token_weights_gp = tf.cast(tf.expand_dims(tf.transpose(token_weights3D, [0, 2, 1]), axis=-1) * tf.expand_dims(
        #     tf.expand_dims(token_weights, axis=1), axis=1), dtype=tf.float32)
        # # abc -> ca, ab
        # token_weights_gp2 = tf.cast(
        #     tf.expand_dims(token_weights3D, axis=2) * tf.expand_dims(tf.expand_dims(token_weights, axis=-1), axis=1),
        #     dtype=tf.float32)
        # token_weights_sib = token_weights_gp
        # token_weights4D = tf.expand_dims(token_weights3D, axis=-3) * tf.expand_dims(tf.expand_dims(token_weights, axis=-1),axis=-1)
        # tokens = {'n_tokens': n_tokens,
        #           'tokens_per_sequence': tokens_per_sequence,
        #           'input_tensors': input_tensors,
        #           'token_weights': token_weights,
        #           'token_weights3D': token_weights3D,
        #           'n_sequences': n_sequences}



        for input_network in input_network_outputs:
            with tf.variable_scope(input_network):
                if input_network == 'GraphParserNetwork':
                    first_order_output = input_network_outputs['GraphParserNetwork']['semgraph']

                    break

        layer = first_order_output['rec_layer']
        tokens = first_order_output['tokens']
        token_weights3D = tokens['token_weights3D']

        unlabeled_predictions = first_order_output['unlabeled_predictions']
        unlabeled_targets = first_order_output['unlabeled_targets']
        unary_logits = first_order_output['label_logits']
        unary_predictions = first_order_output['label_predictions']
        label_targets = first_order_output['label_targets']


        hidden_keep_prob = 1 if reuse else self.hidden_keep_prob
        add_linear = self.add_linear
        hidden_sizes = 2 * self.sib_hidden_size  # sib_head, sib_dep

        with tf.variable_scope('Second_Order'):
            for i in six.moves.range(0, self.n_layers - 1):
                with tf.variable_scope('FC-sib-%d' % i):
                    layer = classifiers.hidden(layer, hidden_sizes,
                                               hidden_func=self.hidden_func,
                                               hidden_keep_prob=hidden_keep_prob)
            with tf.variable_scope('FC-sib-top'):
                layers = classifiers.hiddens(layer, 2 * [self.sib_hidden_size],
                                             hidden_func=self.hidden_func,
                                             hidden_keep_prob=hidden_keep_prob)
            sib_head, sib_dep = layers.pop(0), layers.pop(0)

            unary_logits_shape = nn.get_sizes(unary_logits)  # n, mp, ma, o

            # c_roles = ['A0', 'A1' ,'A2', 'A3', 'AM-TMP', 'AM-MNR', 'AM-ADV', 'AM-LOC', 'AM-DIS', 'AM-EXT']
            c_roles = ['A1']

            output_fields = {vocab.field: vocab for vocab in self.output_vocabs}
            output_vocab = output_fields['semrel']
            # pdb.set_trace()
            idx = []
            for role in c_roles:
                idx.append(TokenVocab.__getitem__(output_vocab,role))
            one_hot_c_roles = tf.one_hot(idx, len(output_vocab))
            unary_logits_l_mask = 1 - tf.reduce_sum(one_hot_c_roles, axis=0)  # 1 x o
            # n x mp x ma x o
            unary_logits_l_roles = unary_logits * tf.expand_dims(
                tf.expand_dims(tf.expand_dims(unary_logits_l_mask, axis=0), axis=0), axis=0)

            # n x mp x ma x 10
            unary_logits_c_roles = tf.matmul(tf.reshape(unary_logits, [-1, len(output_vocab)]),
                                             tf.transpose(one_hot_c_roles))
            unary_logits_c_roles = tf.reshape(unary_logits_c_roles,
                                              [unary_logits_shape[0], unary_logits_shape[1],
                                               unary_logits_shape[2], len(c_roles)])

            mask3D = tf.zeros_like(unary_predictions)
            for i in idx:
                mask3D += tf.where(tf.equal(unary_predictions, i), tf.ones_like(unary_predictions),
                                   tf.zeros_like(unary_predictions))
            mask4D = tf.expand_dims(mask3D, axis=-1)
            token_weights4D = tf.expand_dims(token_weights3D, axis=-1)
            unary_logits_c_roles_mask = unary_logits_c_roles * tf.to_float(mask4D) * tf.to_float(token_weights4D)


            with tf.variable_scope('Classifiers-sib'):
                if self.use_sib:
                    with tf.variable_scope('Sibling'):
                        # layer = (n x 10 x 10 x mp x mb x mc)
                        sib_logits = classifiers.trilinear_classifier(
                            sib_head, sib_dep, sib_dep, len(c_roles),
                            hidden_keep_prob=hidden_keep_prob,
                            add_linear=add_linear)
                        # (n x mp x mb x mc x 10 x 10)
                        sib_logits = tf.transpose(sib_logits, perm=[0, 3, 4, 5, 1, 2])

                q_value = unary_logits_c_roles_mask

                for i in range(int(self.num_iteration)):
                    q_value = tf.nn.softmax(q_value, -1)
                    # ->root 0 ; ->pad-> 0
                    q_value = q_value * tf.to_float(mask4D) * tf.to_float(token_weights4D)
                    # q_value (n x mp x mc x 10) * sib_logits (n x mp x ma x mb x 10 x 10) -> F_temp (n x mp x ma x mb x 10)
                    if self.use_sib:
                        # second_temp_sib = tf.einsum('niac,noiabc->noiab', q_value, layer_sib)
                        F_temp_sib = tf.einsum('npikor,npkr->npiko', sib_logits, q_value)
                        # n x mp x 10 x ma x mb
                        F_temp_sib = tf.transpose(F_temp_sib, perm=[0, 1, 4, 2, 3])
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
                    else:
                        F_temp_sib = 0

                    F = F_temp_sib
                    q_value = unary_logits_c_roles_mask + tf.reduce_sum(F, -2)


            # n x mp x ma x 10: ma masked
            q_value = q_value * tf.to_float(mask4D) * tf.to_float(token_weights4D)
            unary_logits_c_roles_init = unary_logits_c_roles * tf.to_float((1 - mask4D)) * tf.to_float(
                token_weights4D)
            # n x mp x ma x 10
            q_value = q_value + unary_logits_c_roles_init

            # n x mp x ma x 10 -> n x mp x ma x o
            q_value = tf.reshape(tf.matmul(tf.reshape(q_value, [-1, len(c_roles)]), one_hot_c_roles),
                                 [unary_logits_shape[0], unary_logits_shape[1], unary_logits_shape[2], -1])

            # transposed_logits = tf.transpose((unary_logits_l_roles + q_value),perm=[0,2,1,3])
            transposed_logits = unary_logits_l_roles + q_value
        head_probabilities = tf.expand_dims(tf.stop_gradient(first_order_output['head_probabilities']), axis=-1)
        # (n x m x m x c) -> (n x m x m x c)
        label_probabilities = tf.nn.softmax(transposed_logits) * tf.to_float(tf.expand_dims(token_weights3D, axis=-1))
        # (n x m x m), (n x m x m x c), (n x m x m) -> ()
        label_loss = tf.losses.sparse_softmax_cross_entropy(label_targets, transposed_logits,
                                                            weights=token_weights3D * unlabeled_targets)
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
        n_correct_label_sequences = tf.reduce_sum(
            nn.equal(n_correct_label_tokens_per_sequence, n_targets_per_sequence))

        # -----------------------------------------------------------
        # Populate the output dictionary
        rho = self.loss_interpolation

        second_order_output = first_order_output

        second_order_output['label_targets'] = label_targets
        second_order_output['probabilities'] = label_probabilities * head_probabilities
        second_order_output['label_loss'] = label_loss
        # Combination of labeled loss and unlabeled loss
        # first_order_output['loss'] =  2*((1-rho) * outputs['loss'] + rho * label_loss)
        # second_order_output['loss'] = 2 * ((1 - rho) * first_order_output['unlabeled_loss'] + rho * label_loss)
        second_order_output['loss'] = label_loss


        second_order_output['n_true_positives'] = n_true_positives
        second_order_output['n_false_positives'] = n_false_positives
        second_order_output['n_false_negatives'] = n_false_negatives
        second_order_output['n_correct_sequences'] = n_correct_sequences
        second_order_output['n_correct_label_tokens'] = n_correct_label_tokens
        second_order_output['n_correct_label_sequences'] = n_correct_label_sequences


        outputs = {}
        outputs['semgraph'] = first_order_output

        printdata = {}
        printdata['predictions'] = predictions

        printdata['unary_logits_l_roles'] = unary_logits_l_roles
        printdata['unary_logits_c_roles'] = unary_logits_c_roles
        printdata['unary_logits_c_roles_mask'] = unary_logits_c_roles_mask

        printdata['unary_logits_c_roles_init'] = unary_logits_c_roles_init
        printdata['q_value'] = q_value

        printdata['transposed_logits'] = transposed_logits
        printdata['first_order'] = first_order_output
        printdata['label_loss'] = label_loss



        return outputs, tokens, printdata











    # =============================================================

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
    def num_iteration(self):
        return self._config.getint(self, 'num_iteration')

    @property
    def sib_hidden_size(self):
        return self._config.getint(self, 'sib_hidden_size')

    @property
    def use_sib(self):
        return self._config.getboolean(self, 'use_sib')

    @property
    def loss_interpolation(self):
        return self._config.getfloat(self, 'loss_interpolation')

    @property
    def add_linear(self):
        return self._config.getboolean(self, 'add_linear')

    @property
    def hidden_keep_prob(self):
        return self._config.getfloat(self, 'hidden_keep_prob')