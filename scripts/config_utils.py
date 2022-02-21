
import os
import args
import pdb

from collections import defaultdict

def iterfield(configfile, content_type='idx'):
    """
    input lines of config file, output field string and lines of this field
    content_type:  
        'idx', field_contents is as {entry: line_id}, e.g. 'DEFAULT': {'datatype': 6, ...} 
        'value', field_contents is as {entry: value}, e.g. 'DEFAULT': {'datatype': 's0f', ...}
    """
    with open(configfile) as f:
        configlines = f.readlines()
        field = ''
        field_contents = {}

        for line_idx, line in enumerate(configlines):
            line = line.strip()
            if line.startswith('['):
                if field:
                    yield field, field_contents
                    field = ''
                    field_contents = {}
                field = line[1:-1]

            elif field:
                line = line.split('=')  #e.g. ['auto_dir ', ' False'], ['modelname ', ''], ['']
                if len(line)==2:
                    k = line[0].split()[0]
                    if line[1]:
                        v = line[1].split()[0]
                    else:
                        v = ''
                elif len(line) > 2:
                    raise ValueError
                else:
                    k = None
                if k:
                    if content_type=='idx':
                        field_contents[k] = line_idx
                    elif content_type=='value':
                        field_contents[k] = v
        if field:
            yield field, field_contents

def scan_configdir(configdir, scan_start=0):
    """scan configdir, get list of paths of all configfiles"""
    configlist = []
    for root, _, files in os.walk(configdir): # root: configdir的相对路径, e.g.: ./config/SRLSP05/1230
        for file in files[scan_start:]:
            configfilepath = os.path.join(root, file)
            configlist.append(configfilepath)
    return configlist

def config_reader(configfile, content_type='idx'):
    config_dict = defaultdict(dict)
    for field, field_contents in iterfield(configfile, content_type):
        config_dict[field] = field_contents
    return config_dict

def get_config_value(configfile, field, entry):
    config_dict = config_reader(configfile, content_type='value')

    return config_dict[field][entry]

def get_parsetype(datatype):
    if 's6' in datatype:
        parse_type = 'eisner'
    elif 's7' in datatype:
        parse_type = 'dp2x'
    elif 's8' in datatype:
        parse_type = 'dp3w'
    else:
        parse_type = 'dp'
    return parse_type


class Config(object):

    def __init__(self, args, filename):
        self.args = args
        self.filename = filename    # with .cfg
        self.filename_prefix = os.path.splitext(self.filename)[0]
        try:
            self.parse = args.parse
        except:
            self.parse = None
        
        self._filepath = self.gen_filepath()

    def set_gpu(self, gpu):
        self._gpu = gpu

    def set_datatype(self, datatype):
        if datatype:
            self.datatype = datatype
            self.parsetype = get_parsetype(self.datatype)
            self.datadir = os.path.join('data', self.args.treebank, self.datatype)
            self.logpath = os.path.join(self.args.logdir, self.filename_prefix)
            self.savedir = os.path.join('saves', self.args.treebank, self.datatype, self.args.date, self.filename_prefix)
            self.resultdir = os.path.join('TestResults', self.args.date, self.args.treebank, self.filename_prefix)

        else:
            self.datadir = os.path.join('data', self.args.treebank)
            self.logpath = os.path.join(self.args.logdir, self.filename_prefix)
            self.savedir = os.path.join('saves', 'tune', self.filename_prefix, self.args.network_class)
            self.resultdir = os.path.join('TestResults', self.args.date, self.filename_prefix)

        if self.parse == 'greedy':
            self.resultdir = os.path.join(self.resultdir, 'greedy')
        elif self.parse == 'dp':
            self.resultdir = os.path.join(self.resultdir, 'dp')



    def set_modify_contents(self, modify_contents):
        self._modify_contents = modify_contents

    def export_config(self):
        config_dict = config_reader(self.args.template)
        try:
            datatype_ori = get_config_value(self.args.template, 'DEFAULT', 'datatype')
        except:
            datatype_ori = None
        datatype = None
        f = open(self.filepath, 'w')
        with open(self.args.template) as ft:
            outputlines = ft.readlines()[:]
            for field, entry, value in self.modify_contents:
                if field=='#GENERAL#':
                    for field1 in config_dict:
                        if entry in config_dict[field1]:
                            line_idx = config_dict[field1][entry]
                            line = entry + ' = ' + str(value) + '\n'
                            outputlines[line_idx] = line
                else:
                    try:
                        line_idx = config_dict[field][entry]
                        line = entry + ' = ' + str(value) + '\n'
                        outputlines[line_idx] = line
                    except:
                        pdb.set_trace()
                if field=='DEFAULT' and entry=='datatype':
                    datatype = value

        outputlines = ''.join(outputlines)
        f.write(outputlines)
        f.close()
        if datatype:
            self.set_datatype(datatype)
        elif datatype_ori:
            self.set_datatype(datatype_ori)
        else:
            self.set_datatype(None)
        self.gen_commands()







    def gen_filepath(self):
        configdir = os.path.join(args.configdir, args.date)
        return os.path.join(configdir, self.filename)

    def gen_commands(self):
        # if self.args.with_null:
        #     func_dep2eval = 'dep2span_n.py' 
        # else:
        func_dep2eval = 'spandep_eval.py'
        train_command = 'CUDA_VISIBLE_DEVICES={} python main.py train SrlParserNetwork  --noscreen --config_file {} >{}\n'.format(self.device_id, self.filepath, self.logpath)
        
        if self.parse=='greedy':
            parse_id_command = 'CUDA_VISIBLE_DEVICES={} python main.py --save_dir {} run --get_argmax --output_dir {} {}/test.en.id.srl.conllu\n'.format(self.device_id, self.savedir, self.resultdir, self.datadir)
            parse_ood_command = 'CUDA_VISIBLE_DEVICES={} python main.py --save_dir {} run --get_argmax --output_dir {} {}/test.en.ood.srl.conllu\n'.format(self.device_id, self.savedir, self.resultdir, self.datadir)
        elif self.parse=='dp':
            parse_id_command = 'CUDA_VISIBLE_DEVICES={} python main.py --save_dir {} run --parse_type {} --weight_type logits --output_dir {} {}/test.en.id.srl.conllu\n'.format(self.device_id, self.savedir, self.parsetype, self.resultdir, self.datadir)
            parse_ood_command = 'CUDA_VISIBLE_DEVICES={} python main.py --save_dir {} run --parse_type {} --weight_type logits --output_dir {} {}/test.en.ood.srl.conllu\n'.format(self.device_id, self.savedir, self.parsetype, self.resultdir, self.datadir)
        else:
            parse_id_command = 'CUDA_VISIBLE_DEVICES={} python main.py --save_dir {} run --output_dir {} {}/test.en.id.srl.conllu\n'.format(self.device_id, self.savedir, self.resultdir, self.datadir)
            parse_ood_command = 'CUDA_VISIBLE_DEVICES={} python main.py --save_dir {} run --output_dir {} {}/test.en.ood.srl.conllu\n'.format(self.device_id, self.savedir, self.resultdir, self.datadir)
        
        
        dep2span_id_command = 'python scripts/{} {}/test.en.id.srl.conllu {}/test.en.id.srl.conllu {}/test.id.eval\n'.format(func_dep2eval, self.datadir, self.resultdir, self.resultdir)
        dep2span_ood_command = 'python scripts/{} {}/test.en.ood.srl.conllu {}/test.en.ood.srl.conllu {}/test.ood.eval\n'.format(func_dep2eval, self.datadir, self.resultdir, self.resultdir)
        # unlabeleval_id_command = 'python scripts/calc_spans.py  {}/test.en.id.srl.conllu {}/test.en.id.srl.conllu >{}/test.id.unlabel-eval\n'.format(self.datadir, self.resultdir, self.resultdir)
        # unlabeleval_ood_command = 'python scripts/calc_spans.py  {}/test.en.ood.srl.conllu {}/test.en.ood.srl.conllu >{}/test.ood.unlabel-eval'.format(self.datadir, self.resultdir, self.resultdir)

        shell_commands = [train_command, 
                            # parse_id_command, parse_ood_command, 
                            # dep2span_id_command, dep2span_ood_command,
                            # unlabeleval_id_command, unlabeleval_ood_command
                            ]
        self.shell_commands_str = ''.join(shell_commands)

        srleval_id_command = 'scripts/srl-eval.pl data/eval_span/test-wsj.gold {}/test.id.eval >{}/test.id.srl-eval\n'.format(self.resultdir, self.resultdir)
        srleval_ood_command = 'scripts/srl-eval.pl data/eval_span/test-brown.gold {}/test.ood.eval >{}/test.ood.srl-eval\n'.format(self.resultdir, self.resultdir)
        # cat_unlabel_id_command = '#cat {}/test.id.unlabel-eval\n'.format(self.resultdir)
        # cat_unlabel_ood_command = '#cat {}/test.ood.unlabel-eval\n'.format(self.resultdir)
        cat_srleval_id_command = '#cat {}/test.id.srl-eval\n'.format(self.resultdir)
        cat_srleval_ood_command = '#cat {}/test.ood.srl-eval\n'.format(self.resultdir)

        manual_commands = [
                            # srleval_id_command, srleval_ood_command, 
                            # cat_unlabel_id_command, cat_unlabel_ood_command, 
                            # cat_srleval_id_command, cat_srleval_ood_command
                            ]
        self.manual_commands_str = ''.join(manual_commands)        


    @property
    def filepath(self):
        return self._filepath

    @property
    def gpu(self):
        return self._gpu
    @property
    def machine_id(self):
        return int(self.gpu[0])
    @property
    def device_id(self):
        return int(self.gpu[1])
    @property
    def modify_contents(self):
        return self._modify_contents

if __name__=='__main__':
    templatefile = os.path.join(args.configdir, 'template.cfg')

    # ft = open(templatefile)
    # lines = ft.readlines()
    # print(lines[10])
    # ft.close()
    # print('date', args.date)
    # for field, field_contents in iterfield(templatefile):
    #     print(field)
    #     print(field_contents)

    print(config_reader(templatefile))