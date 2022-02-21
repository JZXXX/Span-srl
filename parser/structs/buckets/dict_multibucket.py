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

from .base_multibucket import BaseMultibucket
from .dict_bucket import DictBucket
import pdb

import os
import bert
from bert import tokenization
#***************************************************************
class DictMultibucket(BaseMultibucket, dict):
	""""""

	#=============================================================
	def __init__(self, vocabs, max_buckets=2, config=None):
		""""""

		super(DictMultibucket, self).__init__(max_buckets, config=config)
		dict.__init__(self)

		for vocab in vocabs:
			self[vocab.classname] = [DictBucket(idx, vocab.depth, config=config) for idx in six.moves.range(max_buckets)]
			if vocab.classname == 'SpanheadGraphIndexVocab':
				self.span_vocab = vocab
		self._lengths = []
		self._indices = {vocab.classname: [] for vocab in vocabs}
		self._tokens = {vocab.classname: [] for vocab in vocabs}
		self._max_lengths = []
		running_class = self._config.get('DEFAULT','network_class')
		input_cfg=self._config.get(running_class,'input_vocab_classes')
		# pdb.set_trace()
		#=============================roberta============================
		if 'Bert'.lower() in input_cfg.lower() and 'roberta'.lower() not in input_cfg.lower():
			try:
				is_pretrained = self._config.get('BertVocab', 'use_pretrained_file')=='True'
			except:
				is_pretrained = False
			if is_pretrained:
				self.pretrained_bert=True
			else:
				self.pretrained_bert=False
				bert_path = self._config.get('BertVocab', 'bert_path')
				bert_vocab_file = os.path.join(bert_path,'vocab.txt')
				self.tokenizer = tokenization.FullTokenizer(bert_vocab_file,True)
				self.sep_token=self.tokenizer.convert_tokens_to_ids(['[CLS]','[SEP]'])
		if 'roberta'.lower() in input_cfg.lower():
			try:
				is_pretrained = self._config.get('RoBertaVocab', 'use_pretrained_file')=='True'
			except:
				is_pretrained = False
			if is_pretrained:
				self.pretrained_bert=True
			else:
				from pytorch_transformers import RobertaTokenizer as roberta_tokenization 
				self.pretrained_bert=False
				bert_path = self._config.get('RoBertaVocab', 'bert_path')
				# bert_vocab_file = os.path.join(bert_path,'vocab.txt')
				self.tokenizer= roberta_tokenization.from_pretrained('roberta-large')
				self.sep_token=self.tokenizer.convert_tokens_to_ids(['[CLS]','[SEP]'])



		# output_cfg=self._config.get('GraphParserNetwork','output_vocab_classes')
		self.use_seq2seq=False
		# if 'seq2seq' in output_cfg.lower():
		# 	self.use_seq2seq=True
		return

	#=============================================================
	def reset(self, vocabs):
		""""""

		self._lengths = []
		self._indices = {vocab.classname: [] for vocab in vocabs}
		self._tokens = {vocab.classname: [] for vocab in vocabs}
		for vocab_classname in self:
			for bucket in self[vocab_classname]:
				bucket.reset()
		return

	#=============================================================
	def add(self, indices, tokens, length=0):
		""""""

		assert self._is_open, 'The DictMultibucket is not open for adding entries'

		if length <= 1:
			return
		for vocab_classname in indices.keys():
			self._indices[vocab_classname].append(indices[vocab_classname])
		for vocab_classname in tokens.keys():
			self._tokens[vocab_classname].append(tokens[vocab_classname])
		super(DictMultibucket, self).add(length)
		return

	#=============================================================
	def close(self):
		""""""

		# Decide where everything goes
		self._lengths = np.array(self._lengths)
		self._max_lengths = self.compute_max_lengths(self._lengths, self.max_buckets)
		len2bkt = self.get_len2bkt(self._max_lengths)

		# Open the buckets
		shape = len(self._lengths)

		dtype = [('bucket', 'i4')] + [(vocab_classname, 'i4') for vocab_classname in self.keys()]
		data = np.zeros(shape, dtype=dtype)
		indices_size=len(self._indices[list(self.keys())[0]])
		to_resume=[]
		#pdb.set_trace()
		vocab_names=list(self.keys())
		vocab_names.sort()
		# pdb.set_trace()
		for i, vocab_classname in enumerate(vocab_names):
			# if vocab_classname == 'FormRoBertaVocab':
				# pdb.set_trace
			for bucket in self[vocab_classname]:
				# pdb.set_trace()
				bucket.open()

			# Add sentences to them
			#pdb.set_trace()
			if 'seq2seq' in vocab_classname.lower():
				to_resume.append(vocab_classname)
				continue
			for j, (indices, tokens) in enumerate(zip(self._indices[vocab_classname], self._tokens[vocab_classname])):
				bucket_index = len2bkt[len(indices)]
				sequence_index = self[vocab_classname][bucket_index].add(indices, tokens)
				data[vocab_classname][j] = sequence_index
				if i == 0:
					data['bucket'][j] = bucket_index
				else:
					assert data['bucket'][j] == bucket_index, 'CoNLLU data is somehow misaligned'
		if self.use_seq2seq:
			tar_bucket=data['FormMultivocab']
			for vocab_classname in to_resume:
				data[vocab_classname]=data['FormMultivocab']
		# pdb.set_trace()
		# Close the buckets

		# if 'FlagTokenVocab' in self:
		# 	max_pres = []
		# 	pres2indices = []
		# 	for bucket in self['FlagTokenVocab']:
		# 		# pdb.set_trace()
		# 		bucket_pre = []
		# 		for sent in bucket.indices:
		# 			pre2index = []
		# 			for i in range(len(sent)):
		# 				if sent[i] == 6:
		# 					pre2index.append(i)
		# 			bucket_pre.append(pre2index)
		#
		# 		pres2indices.append(bucket_pre)
		#
		# 		pres_num = list(map(lambda x: x.count(6), bucket.indices))
		# 		max_pres.append(max(pres_num))
		# 	pdb.set_trace()
		# pdb.set_trace()
		for vocab_classname in self:
			#pdb.set_trace()
			for i, bucket in enumerate(self[vocab_classname]):
				if vocab_classname == 'SpanheadGraphIndexVocab' or vocab_classname == 'SpanrelGraphTokenVocab':
					# pdb.set_trace()
					if self.span_vocab.predict_pred:
						bucket.span_close_without_pred()
					else:
						bucket.span_close()
				elif vocab_classname == 'PredIndexVocab':
					bucket.pred_close()
				elif vocab_classname in {'FormBertVocab', "FormRoBertaVocab"}:
					#pdb.set_trace()
					if self.pretrained_bert:
						bucket.bert_close(is_pretrained=True)
					else:
						bucket.bert_close(sep_token=self.sep_token)
				elif vocab_classname=='DepheadBertVocab':
					bucket.bert_close(is_pretrained=True,get_dephead=True)
				elif 'Elmo' in vocab_classname:
					bucket.elmo_close()
				else:
					# if 'seq2seq' in vocab_classname.lower():
					# 	pdb.set_trace()
					bucket.close()
		super(DictMultibucket, self).close(data)

		return

	#=============================================================
	def get_data(self, vocab_classname, indices):
		""""""

		bucket_index = np.unique(self.bucket_indices[indices])
		assert len(bucket_index) == 1, 'Requested data from multiple (or no) buckets'

		bucket_index = bucket_index[0]
		data_indices = self.data[vocab_classname][indices]
		if vocab_classname in {'FormBertVocab', "FormRoBertaVocab"} and not self.pretrained_bert:
			data={}
			for key in self[vocab_classname][bucket_index].data:
				data[key] = self[vocab_classname][bucket_index].data[key][data_indices]
			#pdb.set_trace()
		else:
			data = self[vocab_classname][bucket_index].data[data_indices]
		return data

	#=============================================================
	def get_tokens(self, vocab_classname, indices):
		""""""

		return [self._tokens[vocab_classname][index] for index in indices]

	#=============================================================
	@property
	def lengths(self):
		return self._lengths
	@property
	def max_lengths(self):
		return self._max_lengths
	@property
	def bucket_indices(self):
		return self.data['bucket']
	@property
	def data_indices(self):
		return self.data[:, 2:]
