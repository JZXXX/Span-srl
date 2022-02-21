# How do you run the model?
## environment
- required:
    - python == 3.7
    - cudatoolkit == 10.0.130
    - tensorflow 1.x >=1.13
    - pytorch == 1.2
    - pytorch-transformers
    - bert-serving
    - bert-serving-client
    - numpy == 1.19.5
    - psutil
    - matplotlib
    - scipy

## data
conll 2005 and conll2012 with formation as follow example sentence:
```console
#100002
1	Ms.	_	_	NNP	0	_	_	3:(1,2)-A0	_
2	Haag	_	_	NNP	0	_	_	_	_
3	plays	_	_	VBZ	1	_	_	_	play
4	Elianti	_	_	NNP	0	_	_	3:(4,4)-A1	_
5	.	_	_	.	0	_	_	_	_
```
each line has ten columns,
- column 5: POS; 
- column 6: predicate indicator; 
- column 9: arguments; 
- column 10: predicate sense.

arguments formation: take "3:(1,2)-A0" as an example, it means predicate with id 3 has an argument span contain tokens from id 1 to id 2 and argument label is 'A0'



## config
example configs are in './configs'. config names with 'argument_detection' are for pruning networks and other configs are for SRL network.

## contextualized embeddings
### ELMo
To use elmo embedding, we need to extract token features from elmo model and store it into an HDF5 file as a cache file.
First download option "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json" and weight "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5" and then use './bilm-tf/usage_cached.py' to extract the cache file and place it in the fold './bilm-tf'
### Bert
Please download the bert-large uncased checkpoint, and modify the model path in the config file.
### Roberta
Please download the pretrained pytorch RoBERTa-large checkpoint and follow the instruction of "https://github.com/vickyzayats/roberta_tf_ckpt" to transfer it into tensorflow version and place it in './roberta/tf_ckpt/large'.

## example commands
### train
```console
python3 main.py train SrlParserNetwork  --noscreen --config_file config/conll12/gpred/roberta/firstorder/SRL12.cfg
```
### test
```console
python main.py --save_dir saves/conll12/gpred/roberta/firstorder/SRL12/SrlParserNetwork run --output_dir TestResult/conll12/gpred/roberta/firstorder/SRL12/SrlParserNetwork data/conll12/test.en.srl.conllu
```
### calculate F1
After test step, we can calculate F1 with official script using example commands as follow
With gold predicate:
```console
python scripts/srl_eval.py data/srlconll-1.1/conll12.test.gold TestResult/conll12/gpred/roberta/firstorder/SRL12/SrlParserNetwork/test.en.srl.conllu True
```

With predict predicate:
```console
python scripts/srl_eval.py data/srlconll-1.1/conll12.test.gold TestResult/conll12/ppred/roberta/firstorder/SRL12/SrlParserNetwork/test.en.srl.conllu False
```
For conll05 dataset, please replace 'conll12.test.gold' with 'test.wsj.gold' or 'test.brown.gold'.