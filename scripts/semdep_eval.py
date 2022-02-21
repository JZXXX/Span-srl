#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from collections import Counter, namedtuple
import codecs
import sys

import numpy as np
import pdb

#===============================================================
def compute_F1(gold_files, sys_files, labeled=False):
  """"""
  
  correct = 0
  predicted = 0
  actual = 0
  n_tokens = 0
  n_sequences = 0
  current_seq_correct = False
  n_correct_sequences = 0
  current_fp = 0
  current_sent = 0
  sense_gold = 0
  sense_sys = 0
  sense_correct = 0

  for gold_file, sys_file in zip(gold_files, sys_files):
    with codecs.open(gold_file, encoding='utf-8') as gf,\
         codecs.open(sys_file, encoding='utf-8') as sf:
      gold_line = gf.readline()
      gold_i = 1
      sys_i = 0
      while gold_line:
        while gold_line.startswith('#'):
          current_sent += 1
          gold_i += 1
          n_sequences += 1
          n_correct_sequences += current_seq_correct
          current_seq_correct = True
          gold_line = gf.readline()
        if gold_line.rstrip() != '': 
          sys_line = sf.readline()
          sys_i += 1
          while sys_line.startswith('#') or sys_line.rstrip() == '' or sys_line.split('\t')[0] == '0':
            sys_line = sf.readline()
            sys_i += 1
          
          gold_line = gold_line.rstrip().split('\t')
          sys_line = sys_line.rstrip().split('\t')
          assert sys_line[1] == gold_line[1], 'Files are misaligned at lines {}, {}'.format(gold_i, sys_i)
          
          # Compute the gold edges
          gold_node = gold_line[8]
          if gold_node != '_':
            gold_node = gold_node.split('|')
            if labeled:
              gold_edges = set(tuple(gold_edge.split(':', 1)) for gold_edge in gold_node if not gold_edge.split(':',1)[1]=='None')
            else:
              gold_edges = set(gold_edge.split(':', 1)[0] for gold_edge in gold_node if not gold_edge.split(':',1)[1]=='None')
          else:
            gold_edges = set()
          
          # Compute the sys edges
          sys_node = sys_line[8]
          if sys_node != '_':
            sys_node = sys_node.split('|')
            if labeled:
              sys_edges = set(tuple(sys_edge.split(':', 1)) for sys_edge in sys_node if not sys_edge.split(':',1)[1]=='None')
            else:
              sys_edges = set(sys_edge.split(':', 1)[0] for sys_edge in sys_node if not sys_edge.split(':',1)[1]=='None')
          else:
            sys_edges = set()
          
          correct_edges = gold_edges & sys_edges
          if len(correct_edges) != len(gold_edges):
            current_seq_correct = False
          correct += len(correct_edges)
          predicted += len(sys_edges)
          actual += len(gold_edges)
          n_tokens += 1

          # compute predicate sense....
          gold_sense = []
          sys_sense  = []
          for i in gold_edges:
            if isinstance(i, tuple):
              head = i[0]
            else:
              head = i
            if head == '0':
              sense_gold += 1
              gold_sense.append(i)
              break
          for i in sys_edges:
            if isinstance(i, tuple):
              head = i[0]
            else:
              head = i
            if head == '0':
              sense_sys += 1
              sys_sense.append(i)
              break
          correct_pre = set(gold_sense) & set(sys_sense)
          sense_correct += len(correct_pre)

          
          #current_fp += len(sys_edges) - len(gold_edges & sys_edges)
        gold_line = gf.readline()
        gold_i += 1
  #print(correct, predicted - correct, actual - correct)
  Accuracy = namedtuple('Accuracy', ['sense_precision','sense_recall','sense_F1','precision', 'recall', 'F1', 'seq_acc'])
  precision = correct / (predicted + 1e-12)
  recall = correct / (actual + 1e-12)
  F1 = 2 * precision * recall / (precision + recall + 1e-12)
  seq_acc = n_correct_sequences / n_sequences

  # compute predicate sense...
  sense_precision = sense_correct/(sense_sys+1e-12)
  sense_recall = sense_correct / (sense_gold + 1e-12)
  sense_F1 = 2 * sense_precision * sense_recall / (sense_precision + sense_recall + 1e-12)
  return Accuracy(sense_precision, sense_recall, sense_F1, precision, recall, F1, seq_acc)

# UAS = compute_F1(['test.en.id.srl.conllu'], ['test.en.id.srl.conllu'], labeled=False)
# LAS = compute_F1(['gold.conllu'], ['predict.conllu'], labeled=True)
# print(UAS)
# print(LAS)

#===============================================================
def main():
  """"""

  files = sys.argv[1:]
  n_files = len(files)
  assert (n_files % 2) == 0
  gold_files, sys_files = files[:n_files//2], files[n_files//2:]
  UAS = compute_F1(gold_files, sys_files, labeled=False)
  LAS = compute_F1(gold_files, sys_files, labeled=True)
  #print(UAS.F1, UAS.seq_acc)
  # print('{:0.1f}'.format(LAS.F1*100))
  print('Unlabel score *********************')
  print('{:0.2f}'.format(UAS.precision*100))
  print('{:0.2f}'.format(UAS.recall*100))
  print('{:0.2f}'.format(UAS.F1*100))
  print('Label score **********************')
  print('{:0.2f}'.format(LAS.precision*100))
  print('{:0.2f}'.format(LAS.recall*100))
  print('{:0.2f}'.format(LAS.F1*100))
  print('Sense Unlabel score *********************')
  print('{:0.2f}'.format(UAS.sense_precision * 100))
  print('{:0.2f}'.format(UAS.sense_recall * 100))
  print('{:0.2f}'.format(UAS.sense_F1 * 100))
  print('Sense Label score **********************')
  print('{:0.2f}'.format(LAS.sense_precision * 100))
  print('{:0.2f}'.format(LAS.sense_recall * 100))
  print('{:0.2f}'.format(LAS.sense_F1 * 100))
if __name__ == '__main__':
  main()
