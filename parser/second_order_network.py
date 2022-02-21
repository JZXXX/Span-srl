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
import pdb
from parser.structs import conllu_dataset
from parser.graph_outputs import GraphOutputs, TrainOutputs, DevOutputs
from parser.structs import vocabs


# ***************************************************************
class SecondOrderNetwork(BaseNetwork):
    """"""

    # # =============================================================
    # def __init__(self, input_networks=set(), config=None):
    #     """"""
    #     # pdb.set_trace()
    #     self._config = config
    #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #
    #     self._input_networks = input_networks
    #     input_network_classes = set(input_network.classname for input_network in self._input_networks)
    #     assert input_network_classes == set(
    #         self.input_network_classes), 'Not all input networks were passed in to {}'.format(self.classname)
    #
    #     extant_vocabs = {}
    #     for input_network in self.input_networks:
    #         for vocab in input_network.vocabs:
    #             if vocab.classname in extant_vocabs:
    #                 assert vocab is extant_vocabs[
    #                     vocab.classname], "Two input networks have different instances of {}".format(vocab.classname)
    #             else:
    #                 extant_vocabs[vocab.classname] = vocab
    #
    #     if 'IDIndexVocab' in extant_vocabs:
    #         self._id_vocab = extant_vocabs['IDIndexVocab']
    #     else:
    #         self._id_vocab = vocabs.IDIndexVocab(config=config)
    #         extant_vocabs['IDIndexVocab'] = self._id_vocab
    #
    #     self._input_vocabs = []
    #     self._decoder_vocabs = []
    #     # for input_vocab_classname in self.input_vocab_classes:
    #     #     if input_vocab_classname in extant_vocabs:
    #     #         self._input_vocabs.append(extant_vocabs[input_vocab_classname])
    #     #     else:
    #     #         VocabClass = getattr(vocabs, input_vocab_classname)
    #     #         vocab = VocabClass(config=config)
    #     #         if hasattr(vocab, 'conllu_idx'):
    #     #             # pdb.set_trace()
    #     #             vocab.load() or vocab.count(self.train_conllus)
    #     #             self._input_vocabs.append(vocab)
    #     #         else:
    #     #             # pdb.set_trace()
    #     #             vocab.load() or vocab.count_mrp(self.get_nodes_path)
    #     #             # vocab.count(self.get_nodes_path)
    #     #             self._decoder_vocabs.append(vocab)
    #     #         extant_vocabs[input_vocab_classname] = vocab
    #     #     if 'Bert' in input_vocab_classname:
    #     #         self.use_bert = True
    #     #         self.bertvocab = self._input_vocabs[-1]
    #     #         self.pretrained_bert = self.bertvocab.get_pretrained
    #     #     else:
    #     #         self.use_bert = False
    #     # # pdb.set_trace()
    #     self._output_vocabs = []
    #     self.use_seq2seq = False
    #     # for output_vocab_classname in self.output_vocab_classes:
    #     #     if output_vocab_classname in extant_vocabs:
    #     #         self._output_vocabs.append(extant_vocabs[output_vocab_classname])
    #     #     # if output_vocab_classname == 'FlagTokenVocab' or 'PredicateTokenVocab':
    #     #     # 	continue
    #     #     else:  # create index vocabs and token vocabs (network)
    #     #         VocabClass = getattr(vocabs, output_vocab_classname)
    #     #         vocab = VocabClass(config=config)
    #     #         if hasattr(vocab, 'conllu_idx'):
    #     #             vocab.load() or vocab.count(self.train_conllus)
    #     #         else:
    #     #             vocab.load() or vocab.count(self.get_nodes_path)
    #     #         self._output_vocabs.append(vocab)
    #     #         extant_vocabs[output_vocab_classname] = vocab
    #     if self.use_seq2seq:
    #         # pdb.set_trace()
    #         self._node_id_vocab = vocabs.Seq2SeqIDVocab(config=config)
    #         # self._output_vocabs.append(self._node_id_vocab)
    #         extant_vocabs[self._node_id_vocab.classname] = self._node_id_vocab
    #     self._throughput_vocabs = []
    #     for throughput_vocab_classname in self.throughput_vocab_classes:
    #         if throughput_vocab_classname in extant_vocabs:
    #             self._throughput_vocabs.append(extant_vocabs[throughput_vocab_classname])
    #         else:
    #             VocabClass = getattr(vocabs, throughput_vocab_classname)
    #             vocab = VocabClass(config=config)
    #             if hasattr(vocab, 'conllu_idx'):
    #                 vocab.load() or vocab.count(self.train_conllus)
    #             else:
    #                 vocab.load() or vocab.count(vocab.get_nodes_path)
    #             self._throughput_vocabs.append(vocab)
    #             extant_vocabs[throughput_vocab_classname] = vocab
    #
    #     with tf.variable_scope(self.classname, reuse=False):
    #         self.global_step = tf.Variable(0., trainable=False, name='Global_step')
    #     self._vocabs = set(extant_vocabs.values())
    #     return

    # =============================================================
    
    #====================================================================================
    def build_graph(self, input_network_outputs={}, reuse=True, debug=False, nornn=False):
        """"""


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
            else:
                input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs]

            layer = tf.concat(input_tensors, 2)

        # for input_network in input_network_outputs:
        #     with tf.variable_scope(input_network):
        #         # pdb.set_trace()
        #         # input_tensors.append(input_network.get_input_tensor(output, reuse=reuse))
        #         layer = tf.stop_gradient(input_network_outputs[input_network]['semgraph']['rec_layer'])
        n_nonzero = tf.to_float(tf.count_nonzero(layer, axis=-1, keepdims=True))
        batch_size, bucket_size, input_size = nn.get_sizes(layer)
        layer *= input_size / (n_nonzero + tf.constant(1e-12))

        token_weights = nn.greater(self.id_vocab.placeholder, 0)
        tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
        n_tokens = tf.reduce_sum(tokens_per_sequence)
        n_sequences = tf.count_nonzero(tokens_per_sequence)
        seq_lengths = tokens_per_sequence + 1

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
                    token_weights3D = token_weights3D * (predicate_weights_ex + root_p)
                    break


        tokens = {'n_tokens': n_tokens,
                  'tokens_per_sequence': tokens_per_sequence,
                  'token_weights': token_weights,
                  'token_weights3D': token_weights3D,
                  'n_sequences': n_sequences}

        # ===========================================================================================

        conv_keep_prob = 1. if reuse else self.conv_keep_prob
        recur_keep_prob = 1. if reuse else self.recur_keep_prob
        recur_include_prob = 1. if reuse else self.recur_include_prob

        for i in six.moves.range(self.n_layers):
            conv_width = self.first_layer_conv_width if not i else self.conv_width
            with tf.variable_scope('RNN-{}'.format(i)):
                layer, _ = recurrent.directed_RNN(layer, self.recur_size, seq_lengths,
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

        # pdb.set_trace()
        output_fields = {vocab.field: vocab for vocab in self.output_vocabs}
        outputs = {}
        with tf.variable_scope('Classifiers'):
            if 'semrel' in output_fields:
                vocab = output_fields['semrel']
                    
                with tf.variable_scope('Labeled'):
                    labeled_outputs = vocab.get_trilinear_classifier_new(
                        layer, input_network_outputs['GraphParserNetwork']['semgraph'],
                        tokens,reuse=reuse)
                outputs['semgraph'] = labeled_outputs
                self._evals.add('semgraph')

        # outputs = input_network_outputs['GraphParserNetwork'][0]
        # tokens = input_network_outputs['GraphParserNetwork'][1]
        return outputs, tokens

    # =============================================================
    def parse(self, conllu_files, output_dir=None, output_filename=None, testing=False, debug=False, nornn=False,
                check_iter=False, gen_tree=False, get_argmax=False):
        """"""
        parseset = conllu_dataset.CoNLLUDataset(conllu_files, self.vocabs, config=self._config)

        if output_filename:
            assert len(conllu_files) == 1, "output_filename can only be specified for one input file"
        factored_deptree = None
        factored_semgraph = None
        for vocab in self.output_vocabs:
            if vocab.field == 'deprel':
                factored_deptree = vocab.factorized
            elif vocab.field == 'semrel':
                factored_semgraph = vocab.factorized
        # pdb.set_trace()
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
            # sess.run(tf.variables_initializer(list(non_save_variables)))
            sess.run(tf.global_variables_initializer())
            for first_saver, path in zip(input_network_savers, input_network_paths):
                first_saver.restore(sess, tf.train.latest_checkpoint(path))
                print('First order network load sucessfully!!!!!!!!!')
            saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
            if self.use_bert and not self.pretrained_bert:
                if self.bertvocab.is_training:
                    self.bertvocab.modelRestore(sess, list(bert_variables),
                    #===================================roberta===============================
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