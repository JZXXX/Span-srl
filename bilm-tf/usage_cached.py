'''
ELMo usage example to write biLM embeddings for an entire dataset to
a file.
'''
import pdb
import os
import h5py
from bilm import dump_bilm_embeddings

# Create the dataset file.
setname='SRL05'
dataset_file = setname+'.txt'
idfile=setname+'_ids'+'.txt'
idlist=open(idfile,'r')
idlist=idlist.readlines()
for i in range(len(idlist)):
	idlist[i]=idlist[i].strip()

assert len(idlist)==len(set(idlist)), 'idlist have repeat ids' 
# Location of pretrained LM.  Here we use the test fixtures.
datadir = '.'
vocab_file = os.path.join(datadir, 'vocab-2016-09-10.txt')
options_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
weight_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

# Dump the embeddings to a file. Run this once for your dataset.
embedding_file = setname+'_embeddings'+'.hdf5'
dump_bilm_embeddings(
    vocab_file, dataset_file, options_file, weight_file, embedding_file, idlist=idlist
)
#pdb.set_trace()
# # Load the embeddings from the file -- here the 2nd sentence.
# with h5py.File(embedding_file, 'r') as fin:
#     second_sentence_embeddings = fin['1'][...]

