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
from parser.argument_detection_network import ArgumentDetectionNetwork
from parser.neural import nn, nonlin, embeddings, recurrent, classifiers
from parser.structs.vocabs.pointer_generator import PointerGenerator
import pdb


# ***************************************************************
class SrlParserNetwork(BaseNetwork):
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


        tokens = {'n_tokens': n_tokens,
                  'tokens_per_sequence': tokens_per_sequence,
                  'token_weights': token_weights,
                  'token_weights3D': token_weights3D,
                  'n_sequences': n_sequences}

        use_pred = False
        for input_vocab in self.input_vocabs:
            # pdb.set_trace()
            # if input_vocab.classname == 'SpanheadGraphIndexVocab':
            #     use_pred = input_vocab.predict_pred
            if input_vocab.classname == 'PredIndexVocab':
                use_pred = True
                preds = input_vocab.placeholder
                preds_weights = nn.greater(preds, -1)
                token_weights4D = tf.expand_dims(tf.expand_dims(preds_weights, axis=-1), axis=-1) * tf.expand_dims(token_weights3D,																						   axis=1)
                token_weights4D = tf.linalg.band_part(token_weights4D, 0, -1)
                tokens['preds'] = preds
                break
            # else:
            # 	preds = root_weights
        if not use_pred:
            # token_weights4D = tf.expand_dims(tf.transpose(token_weights3D, [0,2,1]), axis=-2) * tf.expand_dims(
            #     tf.transpose(token_weights3D, [0,2,1]), axis=-1)
            token_weights4D = tf.expand_dims(tf.transpose(token_weights3D, [0, 2, 1]), axis=-1) * tf.expand_dims(
                token_weights3D, axis=-2)
            token_weights4D = tf.linalg.band_part(token_weights4D, 0, -1)
        # preds_weights = nn.greater(preds, -1)
        # token_weights4D = tf.expand_dims(tf.expand_dims(preds_weights,axis=-1),axis=-1) * tf.expand_dims(token_weights3D, axis=1)
        # token_weights4D = tf.linalg.band_part(token_weights4D, 0, -1)

        tokens['token_weights4D'] = token_weights4D


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

            if 'spanrel' in output_fields:
                vocab = output_fields['spanrel']
                head_vocab = output_fields['spanhead']
                # pdb.set_trace()
                if vocab.factorized:
                    with tf.variable_scope('Unlabeled'):
                        if self.pre_sel_args:
                            for input_network in input_network_outputs:
                                with tf.variable_scope(input_network):
                                    # choose the left arguments after prunning
                                    unary_args_score = tf.stop_gradient(input_network_outputs[input_network]['argument']['logits'])
                                    candicate_args_mask = tf.stop_gradient(input_network_outputs[input_network]['argument']['candicate_args_mask'])
                                    token_weights = tf.stop_gradient(input_network_outputs[input_network]['argument']['token_weights'])
                                    unary_args_score_mask = tf.cast((1 - candicate_args_mask * token_weights),
                                                                    tf.float32) * (-1e13)
                                    unary_args_score_trans = unary_args_score + unary_args_score_mask

                                    unary_args_score_trans = tf.reshape(unary_args_score_trans, [batch_size, -1])
                                    sel_num = tf.cast(1. * tf.to_float(bucket_size), tf.int32)
                                    _, ids = tf.math.top_k(unary_args_score_trans, sel_num)
                                    ids_x = tf.cast(ids / bucket_size, tf.int32)
                                    ids_y = ids % bucket_size
                                    ids_sent = tf.tile(tf.expand_dims(tf.range(batch_size), -1), [1, sel_num])
                                    ids_x = tf.reshape(ids_x, [-1, 1])
                                    ids_y = tf.reshape(ids_y, [-1, 1])
                                    ids_sent = tf.reshape(ids_sent, [-1, 1])
                                    top_args = tf.concat([ids_sent, ids_x, ids_y], axis=-1)
                                    top_args_idx = tf.reshape(top_args, [batch_size, sel_num, 3])
                            if use_pred:
                                unlabeled_outputs = head_vocab.get_bilinear_discriminator_with_args(
                                    layer, preds,
                                    token_weights=token_weights4D,
                                    top_args_idx=top_args_idx,
                                    reuse=reuse, debug=debug)
                            else:
                                if self.use_syntax:
                                    syntax_indicator_vocb = output_fields['Semhead']
                                    syntax_label_vocb = output_fields['Semrel']
                                    unlabeled_outputs = head_vocab.get_bilinear_discriminator_with_args_syntax_ppred(
                                        layer,
                                        syntax_indicator_vocb,
                                        syntax_label_vocb,
                                        token_weights=token_weights4D,
                                        top_args_idx=top_args_idx,
                                        reuse=reuse, debug=debug)
                                else:
                                    unlabeled_outputs = head_vocab.get_bilinear_discriminator_with_args_ppred(
                                        layer,
                                        token_weights=token_weights4D,
                                        top_args_idx=top_args_idx,
                                        reuse=reuse, debug=debug)
                        else:
                            if use_pred:
                                unlabeled_outputs = head_vocab.get_bilinear_discriminator(
                                    layer, preds,
                                    token_weights=token_weights4D,
                                    reuse=reuse)

                    with tf.variable_scope('Labeled'):
                        if self.pre_sel_args:
                            if use_pred:
                                labeled_outputs = vocab.get_bilinear_classifier_with_args(
                                    layer, preds, unlabeled_outputs,
                                    token_weights=token_weights4D,
                                    reuse=reuse, debug=debug)
                            else:
                                if self.use_syntax:
                                    labeled_outputs = vocab.get_bilinear_classifier_with_args_syntax_ppred(
                                        layer, unlabeled_outputs,
                                        syntax_indicator_vocb,
                                        syntax_label_vocb,
                                        token_weights=token_weights4D,
                                        reuse=reuse, debug=debug)
                                else:
                                    labeled_outputs = vocab.get_bilinear_classifier_with_args_ppred(
                                        layer, unlabeled_outputs,
                                        token_weights=token_weights4D,
                                        reuse=reuse, debug=debug)
                        else:
                            if use_pred:
                                labeled_outputs = vocab.get_bilinear_classifier(
                                    layer, preds, unlabeled_outputs,
                                    token_weights=token_weights4D,
                                    reuse=reuse)

                else:
                    if self.pre_sel_args:
                        for input_network in input_network_outputs:
                            with tf.variable_scope(input_network):
                                # choose the left arguments after prunning
                                logits = tf.stop_gradient(input_network_outputs[input_network]['argument']['logits'])
                                num_candicate_args = tf.stop_gradient(
                                    input_network_outputs[input_network]['argument']['num_candicate_args'])
                                candicate_args_idx = tf.stop_gradient(
                                    input_network_outputs[input_network]['argument']['candicate_args_idx'])
                                sel_num = tf.cast(self.prune_proportion * tf.to_float(bucket_size), tf.int32)
                                _, idx = tf.math.top_k(logits, k=sel_num)
                                # n x num_cand_args
                                candicate_args_mask = tf.reduce_sum(tf.one_hot(idx, num_candicate_args), axis=-2)
                                top_args_idx = tf.where(candicate_args_mask > 0)
                                top_args = tf.gather_nd(candicate_args_idx, top_args_idx)
                                # n x k x 3   (1,2,3) sentence(1) span(2,3)
                                top_args_idx = tf.reshape(top_args, [batch_size, -1, 3])
                        if use_pred:
                            labeled_outputs = vocab.get_unfactored_bilinear_classifier_with_args_uni(layer, preds,
                                                                                        token_weights=token_weights4D,
                                                                                        top_args_idx=top_args_idx,
                                                                                        reuse=reuse)
                    else:
                        if use_pred:
                            labeled_outputs = vocab.get_unfactored_trilinear_classifier(layer,preds,
                                                                                   token_weights=token_weights3D,
                                                                                   reuse=reuse)
                outputs['spangraph'] = labeled_outputs
                self._evals.add('spangraph')
            elif 'spanhead' in output_fields:
                vocab = output_fields['spanhead']
                outputs['spanhead'] = vocab.get_trilinear_discriminator(
                    layer, preds,
                    token_weights=token_weights4D,
                    reuse=reuse)
                self._evals.add('spanhead')



        return outputs, tokens

    # =============================================================
    def parse(self, conllu_files, output_dir=None, output_filename=None, testing=False, debug=False, nornn=False,
              check_iter=False, gen_tree=False, get_argmax=False):
        """"""

        parseset = conllu_dataset.CoNLLUDataset(conllu_files, self.vocabs,
                                                config=self._config)

        if output_filename:
            assert len(conllu_files) == 1, "output_filename can only be specified for one input file"
        factored_deptree = None
        factored_semgraph = None
        for vocab in self.output_vocabs:
            if vocab.field == 'deprel':
                factored_deptree = vocab.factorized
            elif vocab.field == 'semrel' or vocab.field == 'spanrel':
                factored_semgraph = vocab.factorized
        compare_two = False
        input_network_outputs = {}
        input_network_savers = []
        input_network_paths = []
        for input_network in self.input_networks:
            with tf.variable_scope(input_network.classname, reuse=False):
                input_network_outputs[input_network.classname] = input_network.build_graph(reuse=True)[0]
            network_variables = set(tf.global_variables(scope=input_network.classname))
            first_non_save_variables = set(tf.get_collection('non_save_variables'))
            network_save_variables = network_variables - first_non_save_variables
            first_saver = tf.train.Saver(list(network_save_variables))
            input_network_savers.append(first_saver)
            input_network_paths.append(self._config.getstr(self, input_network.classname + '_dir'))

        with tf.variable_scope(self.classname, reuse=False):
            parse_graph = self.build_graph(input_network_outputs=input_network_outputs, reuse=True, debug=debug, nornn=nornn)
            parse_outputs = DevOutputs(*parse_graph, load=False, factored_deptree=factored_deptree,
                                       factored_semgraph=factored_semgraph, config=self._config)
        # pdb.set_trace()

        all_variables = set(tf.global_variables(scope=self.classname))
        non_save_variables = set(tf.get_collection('non_save_variables'))
        if self.use_bert:
            bert_variables = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='bert'))
        else:
            bert_variables = set()
        save_variables = all_variables - non_save_variables - bert_variables
        saver = tf.train.Saver(list(save_variables), max_to_keep=1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.variables_initializer(list(non_save_variables)))

            for first_saver, path in zip(input_network_savers, input_network_paths):
                first_saver.restore(sess, tf.train.latest_checkpoint(path))

            saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))

            if self.use_bert and not self.pretrained_bert:
                if self.bertvocab.is_training:
                    self.bertvocab.modelRestore(sess, list(bert_variables),
                    #===================================roberta============================
                                                model_dir=os.path.join(self.model_dir, self.bertvocab.name))
                else:
                    self.bertvocab.modelRestore(sess, list(bert_variables))

            parse_outputs.id_buff = parseset.id_buff

            if len(conllu_files) == 1 or output_filename is not None:
                token_weights = parse_graph[1]
                self.parse_file(parseset, token_weights, parse_outputs, sess, output_dir=output_dir,
                                output_filename=output_filename, get_argmax=get_argmax)
            else:
                self.parse_files(parseset, parse_outputs, sess, output_dir=output_dir)

        '''
        parse_scores = sess.run(parse_tensors, feed_dict=feed_dict)
        parse_outputs.update_history(parse_scores)
        parse_outputs.print_recent_history()
        dev_outputs.accuracies
        '''
        return

    # =============================================================
    def parse_file(self, dataset, token_weights, graph_outputs, sess, output_dir=None, output_filename=None,
                   print_time=True, get_argmax=False):
        """"""
        probability_tensors = graph_outputs.probabilities
        logits_tensors = graph_outputs.logits
        try:
            preds = token_weights['preds']
        except:
            preds = tf.constant(0)


        if graph_outputs.prune:
            top_args = graph_outputs.top_args

        input_filename = dataset.conllu_files[0]
        graph_outputs.restart_timer()
        for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
            tokens, lengths = dataset.get_tokens(indices)
            feed_dict = dataset.set_placeholders(indices)

            if graph_outputs.prune:
                probabilities, logits, preds_np, top_args_idx, sents = sess.run(
                    [probability_tensors, logits_tensors, preds, top_args, token_weights['token_weights']],
                    feed_dict=feed_dict)
                # pdb.set_trace()
                bucket_size = sents.shape[-1]
                predictions = graph_outputs.logits_to_preds(probabilities, logits, preds_np, top_args = top_args_idx, bucket_size = bucket_size)

            else:
                probabilities, logits, preds_np = sess.run([probability_tensors, logits_tensors, preds],
                                                           feed_dict=feed_dict)
                predictions = graph_outputs.logits_to_preds(probabilities, logits, preds_np)

            # pdb.set_trace()
            tokens.update({vocab.field: vocab[predictions[vocab.field]] for vocab in self.output_vocabs})

            # try:
            #     tokens.update({vocab.field: vocab[predictions[vocab.field]] for vocab in self.output_vocabs})
            # except:
            #     pdb.set_trace()
            graph_outputs.cache_predictions(tokens, indices)

        if output_dir is None and output_filename is None:
            graph_outputs.print_current_predictions()
        else:
            input_dir, input_filename = os.path.split(input_filename)
            if output_dir is None:
                output_dir = os.path.join(self.save_dir, 'parsed', input_dir)
            elif output_filename is None:
                output_filename = input_filename

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_filename = os.path.join(output_dir, output_filename)
            with codecs.open(output_filename, 'w', encoding='utf-8') as f:
                graph_outputs.dump_current_predictions(f)
        if print_time:
            print('\033[92mParsing 1 file took {:0.1f} seconds\033[0m'.format(time.time() - graph_outputs.time))

        return

    @property
    def syn_weight(self):
        try:
            return self._config.getfloat(self, 'syn_weight')
        except:
            return 1.

    @property
    def prune_proportion(self):
        try:
            return self._config.getfloat(self, 'prune_proportion')
        except:
            return 0.5

    @property
    def use_syntax(self):
        try:
            return self._config.getboolean(self, 'use_syntax')
        except:
            return False

    @property
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

    @property
    def pre_sel_args(self):
        try:
            return self._config.getboolean(self, 'pre_sel_args')
        except:
            return False
