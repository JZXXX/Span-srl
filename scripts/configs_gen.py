import os
import itertools as it
import shutil
from functools import reduce
from config_utils import iterfield, config_reader, get_config_value, Config
from collections import defaultdict

import args
import pdb

def assign_device(gpus, configpathlist):
    """为每个card分配config file"""
    gpuslist = reduce(lambda x,y: x+y, map(lambda x:[(x[0], i) for i in x[1]], zip(gpus.keys(), gpus.values()))) # [(11,0) (11,1), ...]
    n_gpu = len(gpuslist)
    config_num = len(configpathlist)
    
    device_dict = {}  # {machine_idx:{card_idx:[configfile,...],  ...}, ...}
    for machine_idx in gpus:
        device_dict[machine_idx] = defaultdict(list)
    # config_idx: config index waiting for assignment
    for config_idx, configfile in enumerate(configpathlist):
        machine_idx, card_idx = gpuslist[config_idx%n_gpu]
        device_dict[machine_idx][card_idx].append(configfile)

    return device_dict

# def generate_commands(device_id, datadir, configfilepath, logpath, savedir, resultdir):
#     """
#     datadir: e.g. data/SRLSP05/s6f
#     configfilepath: e.g. config/SRLSP05/200326/SP05_s6f_lr0.05_bs10000_spaneval.cfg
#     """
#     configfilename = os.path.split(configfilepath)[1]
#     train_command = 'CUDA_VISIBLE_DEVICES={} python main.py train GraphParserNetwork  --force --noscreen --config_file {} >{}\n'.format(device_id, configfilepath, logpath)
#     parse_id_command = 'CUDA_VISIBLE_DEVICES={} python main.py --save_dir {} run --get_argmax --output_dir {} {}/test.en.id.srl.conllu\n'.format(device_id, savedir, resultdir, datadir)
#     parse_ood_command = 'CUDA_VISIBLE_DEVICES={} python main.py --save_dir {} run --get_argmax --output_dir {} {}/test.en.ood.srl.conllu\n'.format(device_id, savedir, resultdir, datadir)
#     dep2span_id_command = 'python scripts/dep2eval.py {}/test.en.id.srl.conllu {}/test.en.id.srl.conllu {}/test.id.eval\n'.format(datadir, resultdir, resultdir)
#     dep2span_ood_command = 'python scripts/dep2eval.py {}/test.en.ood.srl.conllu {}/test.en.ood.srl.conllu {}/test.ood.eval\n'.format(datadir, resultdir, resultdir)
#     unlabeleval_id_command = 'python scripts/calc_spans.py  {}/test.en.id.srl.conllu {}/test.en.id.srl.conllu >{}/test.id.unlabel-eval\n'.format(datadir, resultdir, resultdir)
#     unlabeleval_ood_command = 'python scripts/calc_spans.py  {}/test.en.ood.srl.conllu {}/test.en.ood.srl.conllu >{}/test.ood.unlabel-eval'.format(datadir, resultdir, resultdir)
#     shell_commands = [train_command, parse_id_command, parse_ood_command, 
#                         dep2span_id_command, dep2span_ood_command,
#                         unlabeleval_id_command, unlabeleval_ood_command]
#     shell_commands_str = ''.join(shell_commands)
#     srleval_id_command = 'scripts/srl-eval.pl data/eval_span/test-wsj.gold {}/test.id.eval >{}/test.id.srl-eval\n'.format(resultdir, resultdir)
#     srleval_ood_command = 'scripts/srl-eval.pl data/eval_span/test-brown.gold {}/test.ood.eval >{}/test.ood.srl-eval\n'.format(resultdir, resultdir)
#     cat_unlabel_id_command = '#cat {}/test.id.unlabel-eval\n'.format(resultdir)
#     cat_unlabel_ood_command = '#cat {}/test.ood.unlabel-eval\n'.format(resultdir)
#     cat_srleval_id_command = '#cat {}/test.id.srl-eval\n'.format(resultdir)
#     cat_srleval_ood_command = '#cat {}/test.ood.srl-eval\n'.format(resultdir)

#     manual_commands = [srleval_id_command, srleval_ood_command, cat_unlabel_id_command, 
#                         cat_unlabel_ood_command, cat_srleval_id_command, cat_srleval_ood_command]
#     manual_commands_str = ''.join(manual_commands)
#     return shell_commands_str, manual_commands_str

# def export_shell(args, device_dict, force=False):
#     shelldir = os.path.join( 'shell', args.date)
#     if os.path.exists(shelldir) and force:
#         shutil.rmtree(shelldir)
#     if not os.path.exists(shelldir):
#         os.mkdir(shelldir)

    
#     sh_command = ''
#     for machine_idx in device_dict:
#         machine_str = 'gpu'+str(machine_idx)
#         for card_idx in device_dict[machine_idx]:
#             card_str = 'card' + str(card_idx)
#             shname = machine_str + '_' + card_str + '.sh'
#             shell_commands = ''
#             shpath = os.path.join(shelldir, shname)
#             configfilepaths = device_dict[machine_idx][card_idx]
#             for configfilepath in configfilepaths:
#                 configfilename = os.path.split(configfilepath)[1][:-4]
#                 datatype = get_config_value(configfilepath, 'DEFAULT', 'datatype')
#                 datadir = os.path.join('data', args.treebank, datatype)
#                 logpath = os.path.join(args.logdir, configfilename+'.log')
#                 # print('logpath', logpath)
#                 savedir = os.path.join('saves', args.treebank, datatype, args.date, configfilename)
#                 resultdir = os.path.join('TestResults', args.date, args.treebank, configfilename)
#                 shell_commands_str, manual_commands_str = generate_commands(card_idx, datadir, configfilepath, logpath, savedir, resultdir)
#                 shell_commands += '# ' + args.date + ' ' + configfilename + '\n'
#                 shell_commands += shell_commands_str + '\n'
#                 shell_commands += manual_commands_str + '\n'
#             # shell_commands += 'hold{}'.format(card_idx) + '\n'
#             with open(shpath, 'w') as f:
#                 f.write(shell_commands)

#             sh_command += 'nohup sh shell/{}/{}>1&\n'.format(args.date, shname) + '\n'

#     sh_command_file = os.path.join(shelldir, 'shcommand')
#     with open(sh_command_file, 'w') as f:
#         # sh_command = 'nohup sh shell/{}>1&\n'.format(shname)
#         f.write(sh_command)
#     return

# def export_config(templatefile, config_dict, modify_contents, saveconfigfile):
#     """
#     config_dict: e.g. {'DEFAULT': {'lang ': 2, 'lc ': 3, ...}, 'Config': {}, 'BaseNetwork': {'n_passes ': 28, ...}, ...}
#         key: field, value: dict, {$entry:line_index}.
#     modify_contents: e.g. 
#         [['DEFAULT', 'datatype', 's6f'], ['BaseNetwork', 'spaneval', True], ['GraphParserNetwork', 'label_mask', False], ['SemrelGraphTokenVocab', 'loss_interpolation', 0.1], ['Optimizer', 'learning_rate', 0.005]]
#     """
#     filename = os.path.split(saveconfigfile)[-1]
#     f = open(saveconfigfile, 'w')
    
#     with open(templatefile) as ft:
#         outputlines = ft.readlines()[:]
#         for field, entry, value in modify_contents:
#             line_idx = config_dict[field][entry]
#             line = entry + ' = ' + str(value) + '\n'
#             outputlines[line_idx] = line
#     outputlines = ''.join(outputlines)
#     f.write(outputlines)
#     f.close()
#     return

# def generate_configs(args):

#     templatefile = os.path.join(args.configdir, 'template.cfg')
#     config_dict = config_reader(templatefile)
#     configdir = os.path.join(args.configdir, args.date)
#     print('save config in {}', configdir)
#     if not os.path.exists(configdir):
#         os.mkdir(configdir)
#     abbr2para = {'datatype': 'datatype',
#                 'lr': 'learning_rate',
#                 'bs':  'batch_size',
#                 'spaneval':'spaneval',
#                 'labelmask':'label_mask',
#                 'factorized':'factorized',
#                 'li':'loss_interpolation'
#                 }
#     para2abbr = dict(zip(list(abbr2para.values()), list(abbr2para.keys())))
#     abbr_exclude = ['datatype']
#     paras_expand = {}
#     for fields, values in args.parameters.items():
#         entries, values = map(list, (values.keys(), values.values()))
#         entry = entries[0]
#         value = values[0]
#         fieldlist = fields.split(':')
#         for field in fieldlist:
#             key = field+':'+entry
#             value = value
#             paras_expand[key] = value

#     # e.g. keys: ['DEFAULT:datatype', 'BaseNetwork:spaneval', 'Optimizer:learning_rate']
#     # e.g. values: [['s0f+', 's7f+'], [True], [0.05]]
#     keys, values = map(list, (paras_expand.keys(), paras_expand.values()))  
    
#     if args.product:
#         uniq_paracomb_list = list(it.product(*values))   # e.g. [('s0f+', True, 0.05), ('s7f+', True, 0.05)]     
#     else:
#         n_keys = len(keys)
#         n_para = len(values[0])
#         assert reduce(lambda x,y:x+y, map(len, values))==n_keys*n_para, 'length of parameters not equal'
#         uniq_paracomb_list = list(zip(*values)) 

#     configpathlist = []
#     for paras in uniq_paracomb_list:
#         abbrkeys = []
#         for key in keys:        # a key contains field:entry
#             key = key.split(':')
#             abbrkeys.append(key[1])
#         paracomb = dict(zip(abbrkeys, paras))
#         filename = args.prefix
#         for key in abbrkeys:
#             if key in para2abbr:
#                 # print('key', key)
#                 if key not in abbr_exclude:
#                     abbr = para2abbr[key]
#                     value = paracomb[key]
#                     filename = filename + '_' + abbr + str(value)
#                 else:
#                     for key in abbr_exclude:
#                         abbr = ''
#                         value = paracomb[key]
#                         filename = filename +'_' + abbr + str(value)

#         saveconfigfile = os.path.join(configdir,  filename + '.cfg')
#         configpathlist.append(saveconfigfile)
#         modify_contents = []
#         for key in keys:
#             field, entry = key.split(':')
#             modify_contents.append([field, entry])
#         for idx, value in enumerate(paras):
#             modify_contents[idx].append(value)
#         modify_contents.append(['DEFAULT', 'date', args.date])
#         modify_contents.append(['DEFAULT', 'save_dir', '${save_metadir}/' + filename])

        
#         export_config(templatefile, config_dict, modify_contents, saveconfigfile)
#     return configpathlist


# def generate_shells(args, configpathlist, force=False):

#     device_dict = assign_device(args.gpus, configpathlist)
#     export_shell(args, device_dict, force)
#     return




class config_generator(object):

    def __init__(self, args):
        self.args = args
        self.uniq_paras = args.unique_parameters
        self.general_paras = args.general_parameters
        # self.parameters = args.parameters
        self.gpus = args.gpus
        self.configdir = os.path.join(args.configdir, args.date)
        print('save config in {}'.format(self.configdir))
        if not os.path.exists(self.configdir):
            os.mkdir(self.configdir)        
        self.gpu_assignment = {}
        self._gpulist = self.gen_gpulist()
        
    def gen_gpulist(self):
        """
        gpulist: e.g. [(25,0), (25,1), ...]  formation (machine_id, device_id*(card id))
        """
        gpulist = []
        for machine_idx in self.gpus:
            self.gpu_assignment[machine_idx] = defaultdict(list)
            for card_idx in self.gpus[machine_idx]:
                gpulist.append((machine_idx, card_idx))
        return gpulist


    def generate_configs(self):

        abbr2para = {'datatype': 'datatype',
                    'lr': 'learning_rate',
                    'decay':'decay_rate',
                    'bs':  'batch_size',
                    'maxstep':'max_steps',
                    'spaneval':'spaneval',
                    'testlr':'test_lr',
                    'lrmax':'lr_max',
                    'lrend':'lr_end',
                    'labelmask':'label_mask',
                    'mask':'mask',
                    'nblock':'num_blocks',
                    'postlstm':'post_lstm',
                    'warmup':'warmup',
                    'warmupsteps':'warmup_steps',
                    'lstm':'use_lstm',
                    'attention':'use_attention',
                    'ff':'use_ff',
                    'noroot':'no_root',
                    'factorized': 'factorized',
                    'li': 'loss_interpolation',
                    'lla': 'labeled_logit_adjustment',
                    'rkp': 'recur_keep_prob',
                    'ekp': 'embed_keep_prob',
                    'pp': 'prune_proportion',
                    'hkp': 'hidden_keep_prob',
                    'sd': 'span_diff',
                    'rt': 'role_tensor',
                    'minoc': 'min_occur_count',
                    'char':'use_subtoken_vocab',
                     'form':'use_token_vocab',
                     'coparent':'co_parent',
                     'hf':'hidden_func',
                     'nlayer':'n_layers'
                    }
        para2abbr = dict(zip(list(abbr2para.values()), list(abbr2para.keys())))
        no_abbr = ['prior_file']
        abbr_exclude = ['datatype'] # config name 中不包含参数缩写，只保留参数值

        #---------------------------------------------------
        # uniq parameters
        paras_expand = {}
        for fields, values in self.uniq_paras.items():  # e.g. fields: 'DEFAULT', 'BaseNetwork', ... values: {'datatype': ['s0f+', 's7f+']}, ...
            entries = list(values.keys())
            fieldlist = fields.split(':')   # e.g. fields: 'CoNLLUTrainset:CoNLLUDevset'
            for field in fieldlist:
                for entry in entries:
                    key = '{}:{}'.format(field, entry)  # e.g.: 'BaseNetwork:spaneval'
                    value = self.uniq_paras[fields][entry]
                    paras_expand[key] = value     # e.g. {'BaseNetwork:spaneval':[True, False], ...}
        
        for key, value in self.general_paras.items():   # e.g. keys: 'batch_size',  ...,  values: [3000, 5000], ...
            key = '#GENERAL#:{}'.format(key)     # '#GENERAL#:batch_size'
            paras_expand[key] = value



        # pdb.set_trace()
        keys, values = map(list, (paras_expand.keys(), paras_expand.values()))  #e.g. keys: ['DEFAULT:datatype', 'BaseNetwork:spaneval', 'Optimizer:learning_rate', '#GENERAL#:batch_size'], values: [['s0f+', 's7f+'], [True], [0.05]]
        # pdb.set_trace()
        if args.product:
            paracomb_list = list(it.product(*values))   # [('s0f+', True, 0.05), ('s7f+', True, 0.05)]
        else:   # values各项必须长度一样。
            n_keys = len(keys)
            n_paras = map(len, values)
            s = set(n_paras)
            assert len(s) in {1,2}
            if len(s) ==1: # values中每个value里的参数个数相同
                assert reduce(lambda x,y:x+y, map(len, values))==n_keys*n_para, 'length of parameters not equal'
                paracomb_list = list(zip(*values)) 
            else:   # 有两种参数个数的话
                assert 1 in s #必须有一种为1个参数
                s.remove(1)
                x = s.pop()
                for i in range(len(values)):
                    if len(values[i])==1:
                        values[i] = values[i]*x
                paracomb_list = list(zip(*values)) 



        # print('paracomb_list', paracomb_list)
        # generate configs
        assigned_gpu = 0
        self.configs = []

        count = 0
        for paras in paracomb_list:
            abbrkeys = []
            for key in keys:
                key = key.split(':')
                abbrkeys.append(key[1])

            paracomb = dict(zip(abbrkeys, paras))
            filename = args.prefix
            for key in abbrkeys:
                if key in para2abbr:
                    # print('key', key)
                    if key not in abbr_exclude:
                        abbr = para2abbr[key]
                        value = paracomb[key]
                        # if value != False :
                        filename = filename + '_' + abbr + str(value)
                    else:
                        for key in abbr_exclude:
                            abbr = ''
                            value = paracomb[key]
                            filename = filename +'_' + abbr + str(value)


            
            # print('filename 0',filename)
            second_order_flag = False
            modify_contents = []    # the element of modify_contents is [field, entry, value].
            for key in keys:
                field, entry = key.split(':')
                modify_contents.append([field, entry])  # [[$field, $entry], ... [$field, $entry]]
                
            for idx, value in enumerate(paras):
                modify_contents[idx].append(value)  #[[$field, $entry, $value], ...]
                if modify_contents[idx] == ['GraphParserNetwork', 'output_vocab_classes', 'SecondOrderGraphIndexVocab:SemrelGraphTokenVocab']:
                    second_order_flag = True
            modify_contents.append(['DEFAULT', 'date', args.date])
            # if second_order_flag:
            #     args.suffix = args.second_suffix
            # else:
            #     args.suffix = args.norm_suffix
            filename += args.suffix



            
            for i in range(self.args.repeat):
                
                if i>0:
                    filename_i = '{}_{}.cfg'.format(filename, i)
                else:
                    filename_i = '{}.cfg'.format(filename)
                print('filename {}, {}'.format(i, filename_i))
                # self.generate_logdir(filename_i)
                modify_contents.append(['DEFAULT', 'save_dir', '${save_metadir}/' + os.path.splitext(filename_i)[0] + '/'])
                config = Config(args, filename_i)
                assigned_gpu_id = assigned_gpu%len(self.gpulist)
                config.set_gpu(self.gpulist[assigned_gpu_id])
                gpu = self.gpulist[assigned_gpu_id]
                self.gpu_assignment[gpu[0]][gpu[1]].append(config)
                assigned_gpu += 1
                config.set_modify_contents(modify_contents)
                config.export_config()
                self.configs.append(config)
                count+=1
        print('generated {} configs'.format(count))
        return



    def generate_shells(self):
        shelldir = self.args.shelldir
        if os.path.exists(shelldir) and self.args.force:
            shutil.rmtree(shelldir)
        if not os.path.exists(shelldir):
            os.mkdir(shelldir)
        
        sh_command = ''
        for machine_idx in self.gpu_assignment:
            machine_str = 'gpu'+str(machine_idx)
            for card_idx in self.gpu_assignment[machine_idx]:
                card_str = 'card' + str(card_idx)
                shname = machine_str + '_' + card_str + '.sh'
                shell_commands = ''
                shpath = os.path.join(shelldir, shname)
                configs = self.gpu_assignment[machine_idx][card_idx]
                for config in configs:
                    shell_commands_str, manual_commands_str = config.shell_commands_str, config.manual_commands_str
                    shell_commands += '# ' + self.args.date + ' ' + config.filename + '\n'
                    shell_commands += shell_commands_str + '\n'
                    shell_commands += manual_commands_str + '\n'
                # shell_commands += 'hold{}'.format(card_idx) + '\n'
                with open(shpath, 'w') as f:
                    f.write(shell_commands)

                sh_command += 'nohup sh shells/{}/{}>1&\n'.format(self.args.date, shname) + '\n'

        sh_command_file = os.path.join(shelldir, 'shcommand')
        with open(sh_command_file, 'w') as f:
            # sh_command = 'nohup sh shell/{}>1&\n'.format(shname)
            f.write(sh_command)
        return

    def generate_logdir(self, filename):
        filename_prefix = os.path.splitext(filename)[0]
        logfold = os.path.join(self.args.logdir, filename_prefix)
        if not os.path.exists(logfold):
            os.makedirs(logfold)

        


    def process(self):
        self.generate_configs()
        self.generate_shells()

    @property
    def gpulist(self):
        return self._gpulist



if __name__=='__main__':   
    generator = config_generator(args)
    generator.process()

