import os

root = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])

debug = False

treebank = 'SRL052' if not debug else 'toy'

prefix = 'SRL05' if not debug else 'debug'
second_order = False

date = '20201209-first-identity'
date = date if not debug else date +'_toy'
# template = os.path.join('config', treebank, 'template.cfg') if not debug else os.path.join('config', treebank, 'template_test.cfg')
configdir = os.path.join('config', treebank)
savedir = os.path.join('saves', treebank)
logdir = './log/{}'.format(date)
shelldir = './shells/{}'.format(date)
folds = [savedir, logdir, configdir,shelldir]
for fold in folds:
	if not os.path.exists(fold):
		os.makedirs(fold)

force = True

product = True
repeat = 1
network_class = 'SrlParserNetwork'
sent_length = 70
train_conllus = 'data/{}/train.en.srl.{}.conllu'.format(treebank, sent_length)
span_diff = False
unique_parameters = {
					'DEFAULT':{'train_conllus': [train_conllus]},

					'BaseNetwork':{'recur_keep_prob':[

													  0.5,
													  0.6,
													  # 0.75,
													  ],
									},
					'SrlParserNetwork':{
									'input_vocab_classes':['FormMultivocab:ElmoVocab:PredIndexVocab:PredicateTokenVocab:FlagTokenVocab'],
									'prune_proportion':[1,0.8],

										},
					'FormMultivocab':{
									'use_subtoken_vocab':[True],
									'use_token_vocab':[True,False],
									'use_pretrained_vocab': [True],
									'embed_keep_prob':[0.67],
									},
					'FormTokenVocab': {
									  'embed_keep_prob':[0.67],
									   'min_occur_count':[ 5 ],
									   },
					'PretrainedVocab': {
						# 'embed_keep_prob':[0.67],
									   'linear_size':[300
														  ],
									   },
					'SpanheadGraphIndexVocab':{
										# 'co_parent':[True,False],
										'hidden_keep_prob': [
												0.5,
																0.8,
																 ],
						'n_layers':[1],
						'hidden_func' : ['identity']
										},

					'SpanrelGraphTokenVocab':{
											'loss_interpolation':[
																# 0.15,
																0.2,
																# 0.3,
																],
											'hidden_keep_prob': [
												# 0.5,
																0.5,
																 ],
											# 'span_diff':[span_diff],
											'role_tensor':[True,False],
											'n_layers':[1],
											'hidden_func' : ['identity']
											},

					'Optimizer':{
								 'learning_rate':[
													# 0.05,
													0.005,
													0.001
								 ],
								#  'decay_rate':[0.8]
								 },
					'ElmoVocab':{
								 'embed_keep_prob':[
													# 0.1,
													# 0.5,
													0.67
								 ],
								 'linear_size':[1024]
								 },
					'FormPretrainedVocab':{'pretrained_file':[
																'data/glove_vecs/glove.840B.300d.05.filtered'],
										'vocab_loadname' : ['${save_metadir}/${LANG}/glove.840B.300d.05.filtered.pkl']
											}
					# 'Optimizer':{'learning_rate':[1e-4]}
					}



general_parameters = {'batch_size':[10000]}
second_suffix = '_2nd'
norm_suffix = '_sl{}_formmulti_840B'.format(sent_length)
if not second_order:
	unique_parameters['SpanheadGraphIndexVocab'].update({'second_order':[False]})
	suffix = norm_suffix
	# template = os.path.join('config', 'template', 'SRL05-argument_srl_first-bs10000_rs600_hs600_loss0.2_lr0.1_crole.cfg')
	template = os.path.join('config', 'tune', 'SRL05_rkp0.6_pp0.8_ekp0.67_ekp0.67_minoc5_li0.2_hkp0.5_rtTrue_lr0.01_bs10000_sl70_lre-3_formmulti.cfg')

else:
	unique_parameters['SpanheadGraphIndexVocab'].update({'second_order': [True]})
	suffix = norm_suffix + second_suffix
	# template = os.path.join('config', 'tune', 'SRL05-argument_srl_second-bs10000_rs600_hs600_loss0.2_lr0.1_hkr0.5_std0.25_hk150_iter3.cfg')
	template = os.path.join('config', 'tune', 'SRL05_rkp0.6_pp0.8_ekp0.67_ekp0.67_minoc5_li0.2_hkp0.5_rtTrue_lr0.01_bs10000_sl70_lre-3_formmulti.cfg')

# suffix = second_suffix if 'SecondOrderGraphIndexVocab:SemrelGraphTokenVocab' in unique_parameters.get('GraphParserNetwork', {}).get('output_vocab_classes', []) else norm_suffix
# parameters = {
#                 # 'CoNLLUTrainset:CoNLLUDevset':{'batch_size':[10000]},
#                 }
#-----------------------------------------
gpus = {
			# 28:[0,1,2,3],
			# 27:[0,1,2,3],
			# 26:[0,1,2,3],
			# 25:[0,1,2,3],
			24:[0,1,2,3],
			# 23:[0,1,2,3],
			22:[0,1,2,3],
			# 21:[0,1,2,3],
			# 20:[0,1,2,3],
			19:[0,1,2,3],
			# 18:[0,1,2,3],
			# 17:[0,1,2,3],
			# 16:[0,1,2,3],
			# 15:[0,1,2,3],
			# 14:[0,1,2,3],
			# 13:[0,1,2,3],
			12:[0,1,2,3],
			# 11:[0,1,2,3],
			# 10:[0,1,2,3],
			# 9:[0,1,2,3],
			# 8:[0,1,2,3],
			# 7:[0,1,2,3],
			6:[0,1,2,3],
			5:[0,1,2,3],
			# 4:[0,1, 2,3],
			# 3:[0,1,2,3],
			# 2:[0,1, 2,3],
			# 1:[0,1, 2,3],

			}

#--------------------------------------
# parse
# parse = 'dp'




#  | 5e-4, 1e-3, 5e-3, 1e-2,  5e-2,

# start_id = 0
# n_maxcommands = 7



# # outputmask: 表示输出哪些内容，e.g.: '0111' 输出：
# #     total_train_commands: No;
# #     total_save_commands: Yes;
# #     total_parse_commands: Yes;
# #     total_eval_commands: Yes.
# outputmask = '0111'

# #------------------------------------
# parse_target = ['dev', 'test.id','test.ood']