import json
import os
import pdb
import codecs
def read_jsonlines(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def json2conllu(file_path):
    sents = read_jsonlines(file_path)
    dir_path = os.path.dirname(file_path)
    write_path = file_path + '.conllu'
    if 'train' in file_path:
        id = 100000
    elif 'dev' in file_path:
        id = 200000
    elif 'wsj' in file_path:
        id = 300000
    elif 'brown' in file_path:
        id = 400000
    else:
        id = 300000

    with open(write_path, 'w') as fw:
        for sent in sents:
            id += 1
            fw.write('#'+str(id)+'\n')
            words = [[token] for token in sent['sentences'][0]]
            words_constiuents = [[] for _ in sent['sentences'][0]]
            # pdb.set_trace()
            srl = sent['srl'][0]
            for i in srl:
                predicate, start, end, label = i
                if label == 'V' or label == 'C-V':
                    # continue
                    words[start].append(str(predicate+1)+':('+str(start+1)+','+str(end+1)+')-'+label)
                else:
                    words[start].append(str(predicate+1)+':('+str(start+1)+','+str(end+1)+')-'+label)
                # pdb.set_trace()


            constiuents = sent['constituents'][0]
            for constituent in constiuents:
                cs, ce, label = constituent
                words_constiuents[ce].append(str(cs+1)+':'+label)



            for w_idx in range(len(words)):
                sent_str = ['_'] * 10
                sent_str[0] = str(w_idx+1)
                sent_str[1] = words[w_idx][0]
                # pdb.set_trace()
                if len(words_constiuents[w_idx])>1:
                    consts = []
                    for const in words_constiuents[w_idx]:
                        consts.append(const)
                    sent_str[7]='|'.join(consts)
                if len(words[w_idx])>1:
                    pred_tmp = []
                    for pred in range(1,len(words[w_idx])):
                        pred_tmp.append(words[w_idx][pred])
                    sent_str[8]='|'.join(pred_tmp)
                fw.write('\t'.join(sent_str)+'\n')
            fw.write('\n')

    return

def document_json2conllu(file_path):
    documents = read_jsonlines(file_path)
    dir_path = os.path.dirname(file_path)
    write_path = file_path + '.conllu'
    if 'train' in file_path:
        id = 1000000
    elif 'dev' in file_path:
        id = 2000000
    elif 'wsj' in file_path:
        id = 3000000
    elif 'brown' in file_path:
        id = 4000000
    else:
        id = 3000000

    with open(write_path, 'w') as fw:
        for document in documents:
            pre_token_num = 0
            sents = document['sentences']
            srls = document['srl']
            for sent_id, sent in enumerate(sents):
                id += 1
                fw.write('#' + str(id) + '\n')
                srl = srls[sent_id]
                token_num = len(sent)
                # pdb.set_trace()
                words = [[token] for token in sent]
                for i in srl:
                    predicate, start, end, label = i
                    if label == 'V' or label == 'C-V':
                        continue
                        # words[start - pre_token_num].append(
                        #     str(0) + ':(' + str(start - pre_token_num + 1) + ',' + str(
                        #         end - pre_token_num + 1) + ')-' + label)
                        # words[start - pre_token_num].append(
                        #     str(predicate - pre_token_num + 1) + ':(' + str(start - pre_token_num + 1) + ',' + str(
                        #         end - pre_token_num + 1) + ')-' + label)
                    else:
                        words[start-pre_token_num].append(
                                str(predicate-pre_token_num + 1) + ':(' + str(start-pre_token_num + 1) + ',' + str(end-pre_token_num + 1) + ')-' + label)
                for w_idx in range(len(words)):
                    sent_str = ['_'] * 10
                    sent_str[0] = str(w_idx+1)
                    sent_str[1] = words[w_idx][0]
                    # pdb.set_trace()
                    if len(words[w_idx])>1:
                        pred_tmp = []
                        for pred in range(1,len(words[w_idx])):
                            pred_tmp.append(words[w_idx][pred])
                        sent_str[8]='|'.join(pred_tmp)
                    fw.write('\t'.join(sent_str)+'\n')
                fw.write('\n')
                pre_token_num += token_num


def add_flag_lemma(f1, f2, file_name):
    # f2: file to change, f1: provided file
    current_sent = 0
    write_path = os.path.join(os.path.dirname(f2),file_name)
    with open(write_path, 'w') as fw:
        with codecs.open(f1, encoding='utf-8') as gf, \
                codecs.open(f2, encoding='utf-8') as sf:
            gold_line = gf.readline()
            while gold_line:
                while gold_line.startswith('#'):
                    current_sent += 1
                    gold_line = gf.readline()
                if gold_line.rstrip() != '':
                    sys_line = sf.readline()
                    while sys_line.startswith('#') or sys_line.rstrip() == '' or sys_line.split('\t')[0] == '0':
                        if sys_line.startswith('#'):
                            fw.write(sys_line)
                        sys_line = sf.readline()

                    gold_line = gold_line.rstrip().split('\t')
                    sys_line = sys_line.rstrip().split('\t')
                    # pdb.set_trace()
                    if not gold_line[1] == sys_line[1]:
                        if gold_line[1].startswith('/'):
                            # pdb.set_trace()
                            gold_line[1] = gold_line[1].lstrip('/')
                    assert sys_line[1] == gold_line[1], 'Files are misaligned at sentence {}'.format(current_sent)
                    assert sys_line[8] == gold_line[8], 'Files are misaligned at sentence {}'.format(current_sent)

                    sys_line[4] = gold_line[4]
                    sys_line[5] = gold_line[5]
                    sys_line[9] = gold_line[9]
                    fw.write('\t'.join(sys_line)+'\n')
                elif not gold_line.rstrip():
                    fw.write('\n')

                gold_line = gf.readline()

    return


file_path = 'data/conll05/test_brown.conll05.jsonlines'
json2conllu(file_path)
# add_flag_lemma('data/conllu12/conll12.dev.conll', 'data/conllu12/dev.english.conllu12.jsonlines.conllu','dev.en.srl.conllu')
# document_json2conllu('data/conll12-gold/train.english.v5.jsonlines')
# document_json2conllu('data/conll12-gold/test.english.conllu12.jsonlines')
# document_json2conllu('data/conll12-gold/dev.english.v5.jsonlines')