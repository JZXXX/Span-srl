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

try:
	import cPickle as pkl
except ImportError:
	import pickle as pkl

import curses
import time

import numpy as np
import tensorflow as tf

from parser.neural import nn
from scripts.chuliu_edmonds import chuliu_edmonds_one_root

import pdb


# ***************************************************************
class GraphOutputs(object):
	""""""

	_dataset = None
	# UF1 unlabeled score, OLS labeled score, LF1 total score
	_print_mapping = [('form', 'Form'),
					  ('lemma', 'Lemma'),
					  ('upos', 'UPOS'),
					  ('xpos', 'XPOS'),
					  ('frame', 'UFeat'),
					  ('dephead', 'UAS'),
					  ('deprel', 'OLS'),
					  ('deptree', 'LAS'),
					  ('semhead', 'UF1'),
					  ('semrel', 'OLS'),
					  ('semgraph', 'LF1'),
					  ('spanend', 'UF1'),
					  ('spanhead', 'UF1'),
					  ('spanrel', 'OLS'),
					  ('spangraph', 'LF1'),
					  ('label', 'NF1'),
					  ('argument','UF1')]

	# =============================================================
	def __init__(self, outputs, tokens, load=False, evals=None, factored_deptree=None, factored_semgraph=None,
				 config=None):
		""""""
		try:
			self._preds = tokens['preds']
		except:
			pass
		self.outputs = outputs
		self._factored_deptree = factored_deptree
		self._factored_semgraph = factored_semgraph
		self._config = config
		self._evals = evals or list(outputs.keys())
		self.prune = False
		# self._evals = config.getlist(self, 'evals')
		valid_evals = set([print_map[0] for print_map in self._print_mapping])
		# pdb.set_trace()
		for eval_ in list(self._evals):
			assert eval_ in valid_evals
		# pdb.set_trace()
		# if 'attribute' in outputs:
		#   self._loss=outputs['attribute']['loss']
		# else:
		loss = []
		if self.balance:
			pdb.set_trace()
			for output in outputs:
				if output == 'deptree' or output == 'dephead':
					loss.append(
						tf.where(tf.is_finite(outputs[output]['loss'] * tokens['syn_weight']), outputs[output]['loss'],
								 0.))
				else:
					loss.append(tf.where(tf.is_finite(outputs[output]['loss']), outputs[output]['loss'], 0.))

			self._loss = tf.add_n(loss)
		else:
			# pdb.set_trace()
			self._loss = tf.add_n([tf.where(tf.is_finite(output['loss']), output['loss'], 0.) for output in outputs.values()])

		self._accuracies = {'total': tokens}

		self._probabilities = {}
		self._logits = {}
		self.time = None

		# pdb.set_trace()
		# -----------------------------------------------------------
		for field in outputs:

			self._probabilities[field] = outputs[field]['probabilities']
			self._accuracies[field] = outputs[field]

			# ------------------------------------------------------------
			self._logits[field] = outputs[field]['logits']
			if field == 'semgraph':
				self._logits['semrel'] = outputs[field]['label_predictions']
			if field == 'spangraph':
				try:
					self.top_args = outputs[field]['top_args_idx']
					self.prune = True
				except:
					pass
		# -------------------------------------------------------------

		# -----------------------------------------------------------
		filename = os.path.join(self.save_dir, '{}.pkl'.format(self.dataset))
		# TODO make a separate History object
		if load and os.path.exists(filename):
			with open(filename, 'rb') as f:
				self.history = pkl.load(f)
		else:
			self.history = {
				'total': {'n_batches': 0,
						  'n_tokens': 0,
						  'n_sequences': 0,
						  'total_time': 0},
				'speed': {'toks/sec': [],
						  'seqs/sec': [],
						  'bats/sec': []}
			}
			for field in self._accuracies:
				if field == 'semgraph':
					for string in ('head', 'graph'):
						self.history['sem' + string] = {
							'loss': [0],
							'tokens': [0],
							'fp_tokens': 0,
							'fn_tokens': 0,
							'sequences': [0]
						}
					if self._factored_semgraph:
						self.history['semrel'] = {
							'loss': [0],
							'tokens': [0],
							'n_edges': 0,
							'sequences': [0]
						}
				elif field == 'argument':
					self.history['argument'] = {
						'loss': [0],
						'tokens': [0],
						'fp_tokens': 0,
						'fn_tokens': 0,
						'sequences': [0]
					}
				elif field == 'spangraph':
					for string in ('head', 'graph'):
						self.history['span' + string] = {
							'loss': [0],
							'tokens': [0],
							'fp_tokens': 0,
							'fn_tokens': 0,
							'sequences': [0]
						}
					if self._factored_semgraph:
						self.history['spanrel'] = {
							'loss': [0],
							'tokens': [0],
							'n_edges': 0,
							'sequences': [0]
						}
				elif field == 'spanhead':
					self.history['spanhead'] = {
						'loss': [0],
						'tokens': [0],
						'fp_tokens': 0,
						'fn_tokens': 0,
						'sequences': [0]
					}
				elif field == 'dephead':
					self.history['dephead'] = {
						'loss': [0],
						'tokens': [0],
						'sequences': [0]
					}
				elif field == 'attribute':
					self.history[field] = {
						'loss': [0],
						'tokens': [0],
						'fp_tokens': 0,
						'fn_tokens': 0,
						'sequences': [0]
					}
				elif field == 'deptree':
					for string in ('head', 'tree'):
						self.history['dep' + string] = {
							'loss': [0],
							'tokens': [0],
							'sequences': [0]
						}
					if self._factored_deptree:
						self.history['deprel'] = {
							'loss': [0],
							'tokens': [0],
							'sequences': [0]
						}
				elif field == 'label':
					self.history[field] = {
						'loss': [0],
						'tokens': [0],
						'n_tokens': [0],
						'sequences': [0]
					}
				elif field not in ('speed', 'total'):
					self.history[field] = {
						'loss': [0],
						'tokens': [0],
						'sequences': [0]
					}
		self.predictions = {'indices': []}
		return

	# ===========================================================
	def logits_to_preds(self, probabilities, logits, lengths, **kwargs):

		predictions = {}
		predictions['semrel'] = sparse_semgraph_preds = []
		predictions['semhead'] = []
		predictions['spanend'] = sparse_spanend_preds = []
		if 'semrel' in logits:
			span_rel = logits['semrel']

		if 'semgraph' in logits and 'spanend' in logits:
			span_sl = logits['semgraph']
			span_sl = span_sl.transpose((0, 2, 1))
			span_el = logits['spanend']
			span_el = span_el.transpose((0, 2, 1))
			# pdb.set_trace()
			sp = span_sl.shape
			n_sent = sp[0]
			for i in range(n_sent):
				sparse_semgraph_preds.append([])
				sparse_spanend_preds.append([])
				sent_l = lengths[i]
				for j in range(sp[1]):
					sparse_semgraph_preds[-1].append([])
					sparse_spanend_preds[-1].append([])
				for pred in range(sent_l):
					if np.all(span_sl[i][pred] == 0):
						continue
					else:
						span_res = self.srl_dp(span_sl[i][pred][1:], span_el[i][pred][1:], sent_l - 1)
						span_res = [0] + span_res
						# pdb.set_trace()
						for arg in range(len(span_res)):
							if span_res[arg] == 1:
								sparse_semgraph_preds[i][arg].append((pred, span_rel[i, arg, pred]))
							elif span_res[arg] == 2:
								sparse_spanend_preds[i][arg].append((pred))

		elif 'spangraph' in logits:
			# (n x p x k x c)
			spangraph_probs = probabilities['spangraph']
			dims = spangraph_probs.shape
			batch_size, pred_nums, arg_nums = dims[0], dims[1], dims[2]
			spanhead_logits = logits['spangraph']  # n x p x k
			if self.prune and 'top_args' in kwargs:
				top_args = kwargs['top_args']  # n x k x 3
				bucket_size = kwargs['bucket_size']
				sent_idx = top_args[:, :, 0]
				arga_idx = top_args[:, :, 1]
				argb_idx = top_args[:, :, 2]
				spanhead_logits = np.transpose(spanhead_logits, [0, 2, 1])  # n x k x p
				# pdb.set_trace()
				spanhead_logits_tmp = np.zeros([batch_size, bucket_size, bucket_size, pred_nums])  # n x m x m x p
				spanhead_logits_tmp[sent_idx, arga_idx, argb_idx] = spanhead_logits
				spanhead_logits = np.transpose(spanhead_logits_tmp, [0, 3, 1, 2])
				# (n x p x m x m x c) -> (n x p x m x m)
				# spanhead_probs = spangraph_probs.sum(axis=-1)
				# (n x p x m x m) -> (n x p x m x m)
				# spanhead_preds = np.where(spanhead_probs >= .5, 1, 0)
				# (n x p x k x c) -> (n x p x k)
				spanrel_preds = np.argmax(spangraph_probs, axis=-1)
				spanrel_preds = np.transpose(spanrel_preds, [0, 2, 1])  # n x k x p
				spanrel_preds_tmp = np.zeros([batch_size, bucket_size, bucket_size, pred_nums])  # n x m x m x p
				spanrel_preds_tmp[sent_idx, arga_idx, argb_idx] = spanrel_preds
				spanrel_preds = np.transpose(spanrel_preds_tmp, [0, 3, 1, 2])
			# (n x p x m x m) (*) (n x p x m x m) -> (n x p x m x m)
			# pdb.set_trace()
			# spangraph_preds = spanhead_preds * spanrel_preds
			else:
				# (n x p x m x m x c) -> (n x p x m x m)
				spangraph_preds = np.argmax(spangraph_probs, axis=-1)

			predictions['spanrel'] = sparse_spangraph_preds = []
			predictions['spanhead'] = []

			for i in range(len(spanhead_logits)):
				sparse_spangraph_preds.append([])
				for spanb_id in range(len(spanhead_logits[i][0])):
					sparse_spangraph_preds[-1].append([])
				for pred_id in range(len(spanhead_logits[i])):
					if type(lengths) is np.ndarray:
						pred_idx = lengths[i][pred_id]
					else:
						pred_idx = pred_id

					# pdb.set_trace()
					spans = self.span_srl_dp(spanhead_logits[i, pred_id])
					if spans:
						for span in spans:
							span_s = span[0]
							span_e = span[1]
							# pdb.set_trace()
							sparse_spangraph_preds[-1][span_s].append(
								(pred_idx, span_e, int(spanrel_preds[i, pred_id, span_s, span_e])))

		return predictions

	# ==========================================================
	def srl_dp(self, b_score, e_score, n):
		res = [0] * n
		stat = {}
		stat['B'] = [0]
		stat['E'] = [1]
		dp = []
		dp.append(b_score[0])
		dp.append(e_score[0])
		for i in range(1, n):
			b_s = [dp[0], b_score[i], dp[0] + b_score[i], dp[1] + b_score[i]]
			e_s = [dp[0] + e_score[i], dp[1]]
			b_ms = max(b_s)
			stb = b_s.index(b_ms)
			stat['B'].append(stb)
			e_ms = max(e_s)
			ste = e_s.index(e_ms)
			stat['E'].append(ste)
			dp[0] = b_ms
			dp[1] = e_ms

		f = [dp[0], dp[1], 0]
		max_score = max(f)
		max_index = f.index(max_score)
		next = ''
		if max_index == 2:
			return res
		elif max_index == 0:
			init = stat['B'][-1]
			if init == 0:
				next = 'B'
			elif init == 1:
				res[-1] = 1
				return res
			elif init == 2:
				next = 'B'
				res[-1] = 1
			else:
				next = 'E'
				res[-1] = 1
		else:
			init = stat['E'][-1]
			if init == 0:
				next = 'B'
				res[-1] = 2
			else:
				next = 'E'

		for i in range(1, n):
			if n - 1 - i == 0:
				if next == 'B':
					res[0] = 1
				else:
					res[0] = 2
				return res
			if next == 'B':
				st = stat['B'][n - 1 - i]
				if st == 0:
					next = 'B'
				elif st == 1:
					res[n - 1 - i] = 1
					return res
				elif st == 2:
					next = 'B'
					res[n - 1 - i] = 1
				else:
					next = 'E'
					res[n - 1 - i] = 1
			else:
				st = stat['E'][n - 1 - i]
				if st == 0:
					next = 'B'
					res[n - 1 - i] = 2
				else:
					next = 'E'

		return res

	# ==========================================================
	def span_srl_dp(self, arg_matrix):
		length = len(arg_matrix)
		spans = []
		tmp = {}
		opt = [0] * (length)
		# print(opt)
		# exit()
		for i in range(1, length):
			max_i = 0
			for j in range(0, i):
				if opt[j] >= max_i or opt[j] + arg_matrix[j + 1][i] >= max_i:
					if opt[j] >= opt[j] + arg_matrix[j + 1][i]:
						max_i = opt[j]
						opt[i] = max_i
						tmp[i] = j
					else:
						max_i = opt[j] + arg_matrix[j + 1][i]
						opt[i] = max_i
						tmp[i] = (j + 1, i)
				else:
					opt[i] = max_i

		token = length - 1
		while token > 0:
			if isinstance(tmp[token], tuple):
				spans.append(tmp[token])
				token = tmp[token][0] - 1
			else:
				token = tmp[token]
		return spans

	# =============================================================
	def probs_to_preds(self, probabilities, lengths, force_MST=False, get_argmax=False):
		""""""

		predictions = {}
		predictions['semrel'] = sparse_semgraph_preds = []
		predictions['semhead'] = []
		if 'form' in probabilities:
			form_probs = probabilities['form']
			if isinstance(form_probs, (tuple, list)):
				form_samples, form_probs = form_probs
				form_preds = np.argmax(form_probs, axis=-1)
				predictions['form'] = form_samples[np.arange(len(form_preds)), form_preds]
			else:
				form_preds = np.argmax(form_probs, axis=-1)
				predictions['form'] = form_preds
		if 'lemma' in probabilities:
			lemma_probs = probabilities['lemma']
			lemma_preds = np.argmax(lemma_probs, axis=-1)
			predictions['lemma'] = lemma_preds
		if 'upos' in probabilities:
			upos_probs = probabilities['upos']
			upos_preds = np.argmax(upos_probs, axis=-1)
			predictions['upos'] = upos_preds
		if 'xpos' in probabilities:
			xpos_probs = probabilities['xpos']
			if isinstance(xpos_probs, (tuple, list)):
				xpos_preds = np.concatenate(
					[np.argmax(xpos_prob_mat, axis=-1)[:, :, None] for xpos_prob_mat in xpos_probs], axis=-1)
			else:
				xpos_preds = np.argmax(xpos_probs, axis=-1)
			predictions['xpos'] = xpos_preds
		if 'frame' in probabilities:
			frame_probs = probabilities['frame']
			frame_preds = np.argmax(frame_probs, axis=-1)
			predictions['frame'] = frame_preds
		# if 'head' in probabilities: # TODO MST algorithms
		#  head_probs = probabilities['head']
		#  head_preds = np.argmax(head_probs, axis=-1)
		#  predictions['head'] = head_preds
		if 'deptree' in probabilities:
			# (n x m x m x c)
			deptree_probs = probabilities['deptree']
			if self._factored_deptree:
				# (n x m x m x c) -> (n x m x m)
				dephead_probs = deptree_probs.sum(axis=-1)
				# (n x m x m) -> (n x m)
				if get_argmax:
					dephead_preds = np.argmax(dephead_probs, axis=-1)
				else:
					dephead_preds = np.zeros(dephead_probs.shape[:2], dtype=np.int32)
					for i, (_dephead_probs, length) in enumerate(zip(dephead_probs, lengths)):
						# print(_dephead_probs)
						# input()
						cle = chuliu_edmonds_one_root(_dephead_probs[:length, :length])
						dephead_preds[i, :length] = cle
				# ()
				bucket_size = dephead_preds.shape[1]
				# (n x m) -> (n x m x m)
				one_hot_dephead_preds = (np.arange(bucket_size) == dephead_preds[..., None]).astype(int)
				# (n x m x m) * (n x m x m x c) -> (n x m x c)
				deprel_probs = np.einsum('ijk,ijkl->ijl', one_hot_dephead_preds, deptree_probs)
				# (n x m x c) -> (n x m)
				deprel_preds = np.argmax(deprel_probs, axis=-1)
			else:
				# (), ()
				bucket_size, n_classes = deptree_probs.shape[-2:]
				# (n x m x m x c) -> (n x m x mc)
				deptree_probs = deptree_probs.reshape([-1, bucket_size, bucket_size * n_classes])
				# (n x m x mc) -> (n x m)
				deptree_preds = np.argmax(deptree_probs, axis=-1)
				# (n x m) -> (n x m)
				dephead_preds = deptree_preds // bucket_size
				deprel_preds = deptree_preds % n_classes
			predictions['dephead'] = dephead_preds
			predictions['deprel'] = deprel_preds
		if 'spangraph' in probabilities:
			# (n x p x m x m x c)
			spangraph_probs = probabilities['spangraph']
			if self._factored_semgraph:
				# (n x p x m x m x c) -> (n x p x m x m)
				spanhead_probs = spangraph_probs.sum(axis=-1)

				# (n x p x m x m) -> (n x p x m x m)
				spanhead_preds = np.where(spanhead_probs >= .5, 1, 0)
				# (n x p x m x m x c) -> (n x p x m x m)
				spanrel_preds = np.argmax(spangraph_probs, axis=-1)
				# (n x p x m x m) (*) (n x p x m x m) -> (n x p x m x m)
				# pdb.set_trace()
				spangraph_preds = spanhead_preds * spanrel_preds
			else:
				# (n x p x m x m x c) -> (n x p x m x m)
				spangraph_preds = np.argmax(spangraph_probs, axis=-1)

			predictions['spanrel'] = sparse_spangraph_preds = []
			predictions['spanhead'] = []

			for i in range(len(spangraph_preds)):
				sparse_spangraph_preds.append([])
				for spanb_id in range(len(spangraph_preds[i][0])):
					sparse_spangraph_preds[-1].append([])
					for pred_id in range(len(spangraph_preds[i])):
						pred_idx = lengths[i][pred_id]
						for spans_id, span in enumerate(spangraph_preds[i][pred_id][spanb_id]):
							if span:
								try:
									sparse_spangraph_preds[-1][-1].append(
										(pred_idx, spans_id, spangraph_preds[i, pred_id, spanb_id, spans_id]))
								except:
									pdb.set_trace()

		if 'semgraph' in probabilities:
			if force_MST:
				# pdb.set_trace()
				deptree_probs = probabilities['semgraph']
				# (n x m x m x c) -> (n x m x m)
				dephead_probs = deptree_probs.sum(axis=-1)
				# (n x m x m) -> (n x m)
				# dephead_preds = np.argmax(dephead_probs, axis=-1)
				dephead_preds = np.zeros(dephead_probs.shape[:2], dtype=np.int32)
				for i, (_dephead_probs, length) in enumerate(zip(dephead_probs, lengths)):
					# print(_dephead_probs)
					# input()
					cle = chuliu_edmonds_one_root(_dephead_probs[:length, :length])
					dephead_preds[i, :length] = cle
				# ()
				bucket_size = dephead_preds.shape[1]
				# (n x m) -> (n x m x m)
				one_hot_dephead_preds = (np.arange(bucket_size) == dephead_preds[..., None]).astype(int)
				# (n x m x m) * (n x m x m x c) -> (n x m x c)
				deprel_probs = np.einsum('ijk,ijkl->ijl', one_hot_dephead_preds, deptree_probs)
				# (n x m x c) -> (n x m)
				deprel_preds = np.argmax(deprel_probs, axis=-1)
				predictions['dephead'] = dephead_preds
				predictions['deprel'] = deprel_preds
			else:
				# pdb.set_trace()
				# (n x m x m x c)
				semgraph_probs = probabilities['semgraph']
				if self._factored_semgraph:
					# (n x m x m x c) -> (n x m x m)
					semhead_probs = semgraph_probs.sum(axis=-1)
					if get_argmax:
						# pdb.set_trace()
						# semhead_preds = np.argmax(semhead_probs,axis=-1)
						semhead_preds = semhead_probs.max(axis=-1, keepdims=1) == semhead_probs
						semhead_preds *= semhead_probs > 0
					else:
						# (n x m x m) -> (n x m x m)
						semhead_preds = np.where(semhead_probs >= .5, 1, 0)
					# (n x m x m x c) -> (n x m x m)
					semrel_preds = np.argmax(semgraph_probs, axis=-1)
					# (n x m x m) (*) (n x m x m) -> (n x m x m)
					# pdb.set_trace()
					semgraph_preds = semhead_preds * semrel_preds
				# pdb.set_trace()
				else:
					# (n x m x m x c) -> (n x m x m)
					semgraph_preds = np.argmax(semgraph_probs, axis=-1)
				predictions['semrel'] = sparse_semgraph_preds = []
				predictions['semhead'] = []
				for i in range(len(semgraph_preds)):
					sparse_semgraph_preds.append([])
					for j in range(len(semgraph_preds[i])):
						sparse_semgraph_preds[-1].append([])
						for k, pred in enumerate(semgraph_preds[i, j]):
							if pred:
								sparse_semgraph_preds[-1][-1].append((k, semgraph_preds[i, j, k]))
		if 'spanend' in probabilities:
			spanend_probs = probabilities['spanend']
			spanend_preds = np.where(spanend_probs >= .5, 1, 0)
			predictions['spanend'] = sparse_semgraph_preds = []
			for i in range(len(spanend_preds)):
				sparse_semgraph_preds.append([])
				for j in range(len(spanend_preds[i])):
					sparse_semgraph_preds[-1].append([])
					for k, pred in enumerate(spanend_preds[i, j]):
						if pred:
							sparse_semgraph_preds[-1][-1].append((k))

		if 'attribute' in probabilities:
			# pdb.set_trace()
			attr_probs = probabilities['attribute']
			attr_preds = np.where(attr_probs >= .5, 1, 0)
			predictions['attr'] = attribute_preds = []
			for i in range(len(attr_preds)):
				attribute_preds.append([])
				for j in range(len(attr_preds[i])):
					attribute_preds[-1].append([])
					for k, pred in enumerate(attr_preds[i, j]):
						if pred:
							attribute_preds[-1][-1].append(k)
		return predictions

	# =============================================================
	def cache_predictions(self, tokens, indices):
		""""""

		self.predictions['indices'].extend(indices)
		for field in tokens:
			if field not in self.predictions:
				self.predictions[field] = []
			self.predictions[field].extend(tokens[field])
		return

	# =============================================================
	def print_current_predictions(self):
		""""""

		order = np.argsort(self.predictions['indices'])
		fields = ['form', 'lemma', 'upos', 'xpos', 'flag', 'dephead', 'deprel', 'semrel', 'attr']
		for i in order:
			j = 1
			token = []
			while j < len(self.predictions['id'][i]):
				token = [self.predictions['id'][i][j]]
				for field in fields:
					if field in self.predictions:
						token.append(self.predictions[field][i][j])
					else:
						token.append('_')
				print(u'\t'.join(token))
				j += 1
			print('')
		self.predictions = {'indices': []}
		return

	# =============================================================
	def dump_current_predictions(self, f):
		""""""

		order = np.argsort(self.predictions['indices'])
		# fields = ['form', 'lemma', 'upos', 'xpos', 'flag', 'dephead', 'deprel', 'semrel', 'attr']
		fields = ['form', 'lemma', 'upos', 'xpos', 'flag', 'dephead', 'semrel', 'spanrel', 'attr']

		for idx, i in enumerate(order):
			j = 1
			token = []
			try:
				f.write(self.id_buff[idx] + '\n')
			except:
				pass

			while j < len(self.predictions['id'][i]):
				token = [self.predictions['id'][i][j]]
				for field in fields:
					if field in self.predictions:
						if field != 'pred':
							token.append(self.predictions[field][i][j])
					else:
						token.append('_')
				f.write('\t'.join(token) + '\n')
				j += 1
			f.write('\n')
		self.predictions = {'indices': []}
		return

	# =============================================================
	def compute_token_accuracy(self, field):
		""""""

		return self.history[field]['tokens'][-1] / (self.history['total']['n_tokens'] + 1e-12)

	# =============================================================
	def compute_node_accuracy(self, field):
		""""""

		return self.history[field]['tokens'][-1] / (self.history[field]['n_tokens'][-1] + 1e-12)

	def compute_token_F1(self, field):
		""""""

		precision = self.history[field]['tokens'][-1] / (
					self.history[field]['tokens'][-1] + self.history[field]['fp_tokens'] + 1e-12)
		# if self.compare_precision:
		#   #print('use precision for comparing model')
		#   #return self.compute_token_accuracy(field)
		#   return precision
		recall = self.history[field]['tokens'][-1] / (
					self.history[field]['tokens'][-1] + self.history[field]['fn_tokens'] + 1e-12)
		return [precision, recall, 2 * (precision * recall) / (precision + recall + 1e-12)]

	def compute_sequence_accuracy(self, field):
		""""""

		return self.history[field]['sequences'][-1] / self.history['total']['n_sequences']

	# =============================================================
	def get_current_accuracy(self):
		""""""

		token_accuracy = 0
		if self.average_acc:
			for field in self.history:
				if field in self.evals:
					if field.startswith('sem'):
						token_accuracy += np.log(self.compute_token_F1(field)[-1] + 1e-12)
					elif field.startswith('span'):
						token_accuracy += np.log(self.compute_token_F1(field)[-1] + 1e-12)
					elif field == 'argument':
						token_accuracy += np.log(self.compute_token_F1(field)[-1] + 1e-12)
					elif field == 'spanend':
						token_accuracy += np.log(self.compute_token_F1(field)[-1] + 1e-12)
					elif field == 'attribute':
						token_accuracy += np.log(self.compute_token_F1(field) + 1e-12)
					elif field == 'label':
						token_accuracy += np.log(self.compute_node_accuracy(field) + 1e-12)
					else:
						token_accuracy += np.log(self.compute_token_accuracy(field) + 1e-12)
			token_accuracy /= len(self.evals)
		else:
			for field in self.history:
				if field in self.evals:
					if field.startswith('sem') or field.startswith('span'):
						token_accuracy += np.log(self.compute_token_F1(field)[-1] + 1e-12)
					token_accuracy += np.log(self.compute_token_F1(field)[-1] + 1e-12)
		return np.exp(token_accuracy) * 100

	# =============================================================
	def get_current_geometric_accuracy(self):
		""""""

		token_accuracy = 0
		for field in self.history:
			if field in self.evals:
				if field.startswith('sem'):
					token_accuracy += np.log(self.compute_token_F1(field) + 1e-12)
				elif field == 'attribute':
					token_accuracy += np.log(self.compute_token_F1(field) + 1e-12)
				elif field == 'label':
					token_accuracy += np.log(self.compute_node_accuracy(field) + 1e-12)
				else:
					token_accuracy += np.log(self.compute_token_accuracy(field) + 1e-12)
		token_accuracy /= len(self.evals)
		return np.exp(token_accuracy) * 100

	# =============================================================
	def restart_timer(self):
		""""""

		self.time = time.time()
		return

	# =============================================================
	def update_history(self, outputs):
		""""""

		self.history['total']['total_time'] += time.time() - self.time
		self.time = None
		self.history['total']['n_batches'] += 1
		self.history['total']['n_tokens'] += outputs['total']['n_tokens']
		self.history['total']['n_sequences'] += outputs['total']['n_sequences']
		for field, output in six.iteritems(outputs):
			# here is how calculate the semrel ...
			# So semhead is unlabeled loss, semgraph is total loss? semrel is the labeled loss
			if field == 'semgraph':
				if self._factored_semgraph:
					self.history['semrel']['loss'][-1] += output['label_loss']
					self.history['semrel']['tokens'][-1] += output['n_correct_label_tokens']
					self.history['semrel']['n_edges'] += output['n_true_positives'] + output['n_false_negatives']
					self.history['semrel']['sequences'][-1] += output['n_correct_label_sequences']
				self.history['semhead']['loss'][-1] += output['unlabeled_loss']
				self.history['semhead']['tokens'][-1] += output['n_unlabeled_true_positives']
				self.history['semhead']['fp_tokens'] += output['n_unlabeled_false_positives']
				self.history['semhead']['fn_tokens'] += output['n_unlabeled_false_negatives']
				self.history['semhead']['sequences'][-1] += output['n_correct_unlabeled_sequences']
				self.history['semgraph']['loss'][-1] += output['loss']
				self.history['semgraph']['tokens'][-1] += output['n_true_positives']
				self.history['semgraph']['fp_tokens'] += output['n_false_positives']
				self.history['semgraph']['fn_tokens'] += output['n_false_negatives']
				self.history['semgraph']['sequences'][-1] += output['n_correct_sequences']
			elif field == 'spangraph':
				if self._factored_semgraph:
					self.history['spanrel']['loss'][-1] += output['label_loss']
					self.history['spanrel']['tokens'][-1] += output['n_correct_label_tokens']
					self.history['spanrel']['n_edges'] += output['n_true_positives'] + output['n_false_negatives']
					self.history['spanrel']['sequences'][-1] += output['n_correct_label_sequences']
				self.history['spanhead']['loss'][-1] += output['unlabeled_loss']
				self.history['spanhead']['tokens'][-1] += output['n_unlabeled_true_positives']
				self.history['spanhead']['fp_tokens'] += output['n_unlabeled_false_positives']
				self.history['spanhead']['fn_tokens'] += output['n_unlabeled_false_negatives']
				self.history['spanhead']['sequences'][-1] += output['n_correct_unlabeled_sequences']
				self.history['spangraph']['loss'][-1] += output['loss']
				self.history['spangraph']['tokens'][-1] += output['n_true_positives']
				self.history['spangraph']['fp_tokens'] += output['n_false_positives']
				self.history['spangraph']['fn_tokens'] += output['n_false_negatives']
				self.history['spangraph']['sequences'][-1] += output['n_correct_sequences']
			elif field == 'argument':
				self.history['argument']['loss'][-1] += output['unlabeled_loss']
				self.history['argument']['tokens'][-1] += output['n_unlabeled_true_positives']
				self.history['argument']['fp_tokens'] += output['n_unlabeled_false_positives']
				self.history['argument']['fn_tokens'] += output['n_unlabeled_false_negatives']
				self.history['argument']['sequences'][-1] += output['n_correct_unlabeled_sequences']
			elif field == 'spanhead':
				self.history['spanhead']['loss'][-1] += output['unlabeled_loss']
				self.history['spanhead']['tokens'][-1] += output['n_unlabeled_true_positives']
				self.history['spanhead']['fp_tokens'] += output['n_unlabeled_false_positives']
				self.history['spanhead']['fn_tokens'] += output['n_unlabeled_false_negatives']
				self.history['spanhead']['sequences'][-1] += output['n_correct_unlabeled_sequences']
			elif field == 'dephead':
				self.history['dephead']['loss'][-1] += output['unlabeled_loss']
				self.history['dephead']['tokens'][-1] += output['n_correct_unlabeled_tokens']
				self.history['dephead']['sequences'][-1] += output['n_correct_unlabeled_sequences']
			elif field == 'attribute':
				# pdb.set_trace()
				self.history[field]['loss'][-1] += output['attribute_loss']
				self.history[field]['tokens'][-1] += output['n_true_positives']
				self.history[field]['fp_tokens'] += output['n_false_positives']
				self.history[field]['fn_tokens'] += output['n_false_negatives']
				self.history[field]['sequences'][-1] += output['n_correct_sequences']
			elif field == 'deptree':
				if self._factored_deptree:
					self.history['deprel']['loss'][-1] += output['label_loss']
					self.history['deprel']['tokens'][-1] += output['n_correct_label_tokens']
					self.history['deprel']['sequences'][-1] += output['n_correct_label_sequences']
				self.history['dephead']['loss'][-1] += output['unlabeled_loss']
				self.history['dephead']['tokens'][-1] += output['n_correct_unlabeled_tokens']
				self.history['dephead']['sequences'][-1] += output['n_correct_unlabeled_sequences']
				self.history['deptree']['loss'][-1] += output['loss']
				self.history['deptree']['tokens'][-1] += output['n_correct_tokens']
				self.history['deptree']['sequences'][-1] += output['n_correct_sequences']
			elif field == 'label':
				self.history[field]['loss'][-1] += output['loss']
				self.history[field]['tokens'][-1] += output['n_correct_tokens']
				self.history[field]['sequences'][-1] += output['n_correct_sequences']
				self.history[field]['n_tokens'][-1] += output['n_tokens']
			elif field != 'total':
				# pdb.set_trace()
				self.history[field]['loss'][-1] += output['loss']
				self.history[field]['tokens'][-1] += output['n_correct_tokens']
				self.history[field]['sequences'][-1] += output['n_correct_sequences']
		return

	# =============================================================
	def print_recent_history(self, stdscr=None, dataprint=False):
		""""""

		n_batches = self.history['total']['n_batches']
		n_tokens = self.history['total']['n_tokens']
		n_sequences = self.history['total']['n_sequences']
		total_time = self.history['total']['total_time']
		self.history['total']['n_batches'] = 0
		self.history['total']['n_tokens'] = 0
		self.history['total']['n_sequences'] = 0
		self.history['total']['total_time'] = 0

		# -----------------------------------------------------------
		if stdscr is not None:
			stdscr.addstr('{:5}\n'.format(self.dataset.title()), curses.color_pair(1) | curses.A_BOLD)
			stdscr.clrtoeol()
		elif dataprint:
			pass
		else:
			print('{:5}\n'.format(self.dataset.title()), end='')
		# semhead semgraph semrel
		for field, string in self._print_mapping:
			if field in self.history:
				pre, recall = 0.0, 0.0
				tokens = self.history[field]['tokens'][-1]
				if field in ('semgraph', 'semhead'):
					tp = self.history[field]['tokens'][-1]
					info = self.compute_token_F1(field)
					pre, recall, f1 = info[0] * 100, info[1] * 100, info[2] * 100
					self.history[field]['tokens'][-1] = f1
				elif field in ('spangraph', 'spanhead'):
					tp = self.history[field]['tokens'][-1]
					info = self.compute_token_F1(field)
					pre, recall, f1 = info[0] * 100, info[1] * 100, info[2] * 100
					self.history[field]['tokens'][-1] = f1
				elif field == 'argument':
					info = self.compute_token_F1(field)
					pre, recall, f1 = info[0] * 100, info[1] * 100, info[2] * 100
					self.history[field]['tokens'][-1] = f1
				elif field == 'attribute':
					tp = self.history[field]['tokens'][-1]
					self.history[field]['tokens'][-1] = self.compute_token_F1(field) * 100
				elif field == 'label':
					tp = self.history[field]['tokens'][-1]
					self.history[field]['tokens'][-1] = self.compute_node_accuracy(field) * 100
				elif field == 'semrel':
					n_edges = self.history[field]['n_edges']
					self.history[field]['tokens'][-1] *= 100 / n_edges
					self.history[field]['n_edges'] = 0
				elif field == 'spanrel':
					n_edges = self.history[field]['n_edges']
					self.history[field]['tokens'][-1] *= 100 / n_edges
					self.history[field]['n_edges'] = 0
				else:
					self.history[field]['tokens'][-1] *= 100 / n_tokens
				self.history[field]['loss'][-1] /= n_batches
				self.history[field]['sequences'][-1] *= 100 / n_sequences
				loss = self.history[field]['loss'][-1]
				acc = self.history[field]['tokens'][-1]
				acc_seq = self.history[field]['sequences'][-1]
				if stdscr is not None:
					stdscr.addstr('{:5}'.format(string), curses.color_pair(6) | curses.A_BOLD)
					stdscr.addstr(' | ')
					stdscr.addstr('Loss: {:.2e}'.format(loss), curses.color_pair(3) | curses.A_BOLD)
					stdscr.addstr(' | ')
					stdscr.addstr('Acc: {:5.2f}'.format(acc), curses.color_pair(4) | curses.A_BOLD)
					stdscr.addstr(' | ')
					stdscr.addstr('Seq: {:5.2f}\n'.format(acc_seq), curses.color_pair(4) | curses.A_BOLD)
					stdscr.clrtoeol()
				elif dataprint:
					# print('{:5}'.format(string), end='\t')
					print('{:5.2f}'.format(acc), end=' ')
				else:
					print('{:5}'.format(string), end='')
					print(' | ', end='')
					print('Loss: {:.2e}'.format(loss), end='')
					print(' | ', end='')
					print('Pre: {:5.2f}'.format(pre), end='')
					print(' | ', end='')
					print('Rec: {:5.2f}'.format(recall), end='')
					print(' | ', end='')
					print('Acc: {:5.2f}'.format(acc), end='')
					print(' | ', end='')
					print('Seq: {:5.2f}\n'.format(acc_seq), end='')
				for key, value in six.iteritems(self.history[field]):
					if hasattr(value, 'append'):
						value.append(0)
					else:
						self.history[field][key] = 0

		self.history['speed']['toks/sec'].append(n_tokens / total_time)
		self.history['speed']['seqs/sec'].append(n_sequences / total_time)
		self.history['speed']['bats/sec'].append(n_batches / total_time)
		tps = self.history['speed']['toks/sec'][-1]
		sps = self.history['speed']['seqs/sec'][-1]
		bps = self.history['speed']['bats/sec'][-1]
		if stdscr is not None:
			stdscr.clrtoeol()
			stdscr.addstr('Speed', curses.color_pair(6) | curses.A_BOLD)
			stdscr.addstr(' | ')
			stdscr.addstr('Seqs/sec: {:6.1f}'.format(sps), curses.color_pair(5) | curses.A_BOLD)
			stdscr.addstr(' | ')
			stdscr.addstr('Bats/sec: {:4.2f}\n'.format(bps), curses.color_pair(5) | curses.A_BOLD)
			stdscr.clrtoeol()
			stdscr.addstr('Count', curses.color_pair(6) | curses.A_BOLD)
			stdscr.addstr(' | ')
			stdscr.addstr('Toks: {:6d}'.format(n_tokens), curses.color_pair(7) | curses.A_BOLD)
			stdscr.addstr(' | ')
			stdscr.addstr('Seqs: {:5d}\n'.format(n_sequences), curses.color_pair(7) | curses.A_BOLD)
		elif dataprint:
			pass
		else:
			print('Speed', end='')
			print(' | ', end='')
			print('Seqs/sec: {:6.1f}'.format(sps), end='')
			print(' | ', end='')
			print('Bats/sec: {:4.2f}\n'.format(bps), end='')
			print('Count', end='')
			print(' | ', end='')
			print('Toks: {:6d}'.format(n_tokens), end='')
			print(' | ', end='')
			print('Seqs: {:5d}\n'.format(n_sequences), end='')
		filename = os.path.join(self.save_dir, '{}.pkl'.format(self.dataset))
		with open(filename, 'wb') as f:
			pkl.dump(self.history, f, protocol=pkl.HIGHEST_PROTOCOL)
		return

	# =============================================================
	@property
	def evals(self):
		return self._evals

	@property
	def accuracies(self):
		return dict(self._accuracies)

	@property
	def probabilities(self):
		return dict(self._probabilities)

	@property
	def logits(self):
		return dict(self._logits)

	@property
	def preds(self):
		return dict(self._preds)

	@property
	def loss(self):
		return self._loss

	@property
	def save_dir(self):
		return self._config.getstr(self, 'save_dir')

	@property
	def compare_precision(self):
		# pdb.set_trace()
		try:
			if self._config.getstr(self, 'tb') == 'ptb' or self._config.getstr(self, 'tb') == 'ctb':
				return True
			else:
				return False
		except:
			return False

	@property
	def dataset(self):
		return self._dataset

	@property
	def get_print_dict(self):
		evals = self.outputs
		printdict = {}
		if 'semgraph' in evals:
			if 'printdata' in evals['semgraph']:
				printdict = evals['semgraph']
			if 'attribute' in evals:
				printdict['attribute'] = evals['attribute']
		if 'deptree' in evals:
			# pdb.set_trace()
			if 'printdata' in evals['deptree']:
				printdict = evals['deptree']
		return printdict

	@property
	def average_acc(self):
		try:
			return self._config.getboolean(self, 'average_acc')
		except:
			return False

	@property
	def balance(self):
		try:
			return self._config.getboolean(self, 'balance')
		except:
			return False


# ***************************************************************
class TrainOutputs(GraphOutputs):
	_dataset = 'train'


class DevOutputs(GraphOutputs):
	_dataset = 'dev'
