import codecs
import os
import sys
from data_process import gen_sentences
import subprocess
import pdb


column_types = ['id', 'form', 'lemma', 'upos', 'xpos', 'pred', 'head', 'deprel', 'srl', 'misc']

def read_gold_predicates(gold_path):
  fin = codecs.open(gold_path, "r", "utf-8")
  gold_predicates = [[],]
  for line in fin:
    line = line.strip()
    if not line:
      gold_predicates.append([])
    else:
      info = line.split()
      gold_predicates[-1].append(info[0])
  fin.close()
  return gold_predicates

def srl_eval(evalfile, gold_path, parse_file, gold_predicate='True'):
    root_path = os.path.split(evalfile)[0]
    if root_path == '':
        root_path = '.'
    tmp_path = os.path.join(root_path, 'tmp')
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    parsefile = os.path.split(parse_file)[-1]
    ex_file = os.path.join(tmp_path, "{}_eval".format(parsefile))

    sent_list = gen_sentences(parse_file, column_types)
    gold_predicates = read_gold_predicates(gold_path)

    with open(ex_file, 'w') as f:
        for sent_id, sent in enumerate(sent_list):
            ex_line = sent.export_conll05(gold_predicates[sent_id])
            f.write(ex_line+'\n')

    # command = 'sh {}/run_conll_eval.sh {} {}'.format(root_path, input1, input2)
    # (status, output) = subprocess.getstatusoutput(command)
    # pdb.set_trace()
    if gold_predicate=='True':
        child = subprocess.Popen('sh {}/run_conll_eval.sh {} {}'.format(
            root_path, gold_path, ex_file), shell=True, stdout=subprocess.PIPE)
        eval_info = child.communicate()[0].decode('utf-8')
        conll_precision = float(eval_info.strip().split("\n")[6].strip().split()[4])
        conll_recall = float(eval_info.strip().split("\n")[6].strip().split()[5])
        conll_f1 = float(eval_info.strip().split("\n")[6].strip().split()[6])
        print(eval_info)
        print("Official CoNLL Precision={}, Recall={}, Fscore={}".format(
            conll_precision, conll_recall, conll_f1))
    else:
        child = subprocess.Popen('sh {}/run_conll_eval.sh {} {}'.format(
            root_path, gold_path, ex_file), shell=True, stdout=subprocess.PIPE)
        eval_info = child.communicate()[0].decode('utf-8')
        conll_recall = float(eval_info.strip().split("\n")[6].strip().split()[5])

        child = subprocess.Popen('sh {}/run_conll_eval.sh {} {}'.format(
            root_path, ex_file, gold_path), shell=True, stdout=subprocess.PIPE)
        eval_info1 = child.communicate()[0].decode('utf-8')
        conll_precision = float(eval_info1.strip().split("\n")[6].strip().split()[5])

        if conll_recall + conll_precision > 0:
            conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision)
        else:
            conll_f1 = 0
        print(eval_info)
        print(eval_info1)
        print("Official CoNLL Precision={}, Recall={}, Fscore={}".format(
            conll_precision, conll_recall, conll_f1))

    return

if __name__=='__main__':
    # evalfile, gold, parse_file = sys.argv[0], sys.argv[1], sys.argv[2]
    # root_path = os.path.split(evalfile)[0]
    # conll_recall = srl_eval(root_path, gold, parse_file)
    # conll_precision = srl_eval(root_path, gold, parse_file, True)
    argvs = sys.argv
    srl_eval(*argvs)
