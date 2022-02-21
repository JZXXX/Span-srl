import os
import pdb
from collections import defaultdict
from typing import List, Dict, Union



class Label:
    """
    This class represents a label of a sentence. Each label has a value and optionally a confidence score. The
    score needs to be between 0.0 and 1.0. Default value for the score is 1.0.
    """

    def __init__(self, value: str):
        self.value = value
        super().__init__()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not value and value != "":
            raise ValueError(
                "Incorrect label value provided. Label value needs to be set."
            )
        else:
            self._value = value

    def __str__(self):
        return "{}".format(self._value)

    def __repr__(self):
        return "{}".format(self._value)

class Token():

    def __init__(
        self,
        text: str = None,
        idx: int = None,
    ):
        self.text: str = text
        self.idx: int = idx
        self.tags: dict = {}

    def set_text(self, text: str):
        self.text = text

    def add_tag(self, tag_type: str, tag_value):
        self.tags[tag_type] = tag_value

    # def add_tag(self, tag_type: str, tag_value: str):   # e.g. tag_type: 'text', 'pos', 'chunk', 'ner', tag_value: 'EU', 'NNP', 'I-NP', 'B-ORG' 

    #     tag = tag_value
    #     self.tags[tag_type] = tag

    def get_tag(self, tag_type: str):
        if tag_type in self.tags:
            return self.tags[tag_type]
        return ""

    def __str__(self) -> str:
        return (
            "Token: {} {}".format(self.idx, self.text)
            if self.idx is not None
            else "Token: {}".format(self.text)
        )

    def __repr__(self) -> str:
        return (
            "Token: {} {}".format(self.idx, self.text)
            if self.idx is not None
            else "Token: {}".format(self.text)
        )

class Span():

    def __init__(self, 
                tokens: List[Token]=None, 
                tag: str = None,
                predicate: Token = None,   
                start_pos: int = None,
                end_pos: int = None,             
                ):   # initial a Span: predicate token and a list of tokens in this span, or give (predicate token, tag, start index and end index of the span tokens)
        self.tokens = tokens
        self.tag = tag
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.predicate = predicate   # predicate token

        if tokens:
            self.start_pos = tokens[0].idx
            self.end_pos = tokens[len(tokens) - 1].idx

    @property
    def text(self) -> str:
        return " ".join([t.text for t in self.tokens])
    @property
    def predicate_id(self) -> str:
        if self.predicate != None:
            return int(self.predicate.idx)
        else:
            return 0

    def __str__(self) -> str:
        ids = '{}-{}'.format(self.start_pos, self.end_pos)
        return (
            '<{}-span, pred:[{}], [{}]: "{}">'.format(self.tag, self.predicate, ids, self.text)
            if self.tag is not None
            else 'span [{}]: "{}"'.format(ids, self.text)
        )

    def __repr__(self) -> str:
        ids = '{}-{}'.format(self.start_pos, self.end_pos)
        return (
            '<{}-span, pred:[{}], [{}]: "{}">'.format(self.tag, self.predicate, ids, self.text)
            if self.tag is not None
            else '<span ({}): "{}">'.format(ids, self.text)
        )
    
    def export_span(self) -> List:
        return([int(self.predicate_id), int(self.start_pos), int(self.end_pos), self.tag])

class Sentence():

    def __init__(
        self,
        sent_id: str = None,
        text: str = None,   
        column_types: list = None,
        # labels: Union[List[Label], List[str]] = None,
        lines: list = None, #list of line strings.
        ):

        # super(Sentence, self).__init__()
        self.sent_id = sent_id
        self.predicates = set() # set of tokens idx
        self.tokens: List[Token] = []
        self.spans: List[Span] = []

        for line in lines:
            # pdb.set_trace()
            split_str = line.split()
            if len(split_str) > 0:
                try:
                    new_token = Token(idx=int(split_str[0]))
                except:
                    pdb.set_trace()
                for column_id, column_type in enumerate(column_types):
                    new_token.add_tag(tag_type=column_type, tag_value=split_str[column_id])
                new_token.set_text(new_token.get_tag('form'))
            else:
                pdb.set_trace()
            self.add_token(new_token)
                   
    def get_token(self, token_id: int) -> Token:
        for token in self.tokens:
            if token.idx == token_id:
                return token

    def add_token(self, token: Union[Token, str]):

        if type(token) is str:
            token = Token(token)

        self.tokens.append(token)

        # set token idx if not set
        token.sentence = self
        if token.idx is None:
            token.idx = len(self.tokens)

    def add_predicate(self, idx):
        self.predicates.add(idx)

    # def get_predicates(self):
        # for token in self.tokens:
        #     if token.get_tag('pred')=='1':
        #         self.add_predicate(token.idx)
        # return self.predicates


    def get_spans(self, tag_type: str='bioes') -> List[Span]:

        spans: List[Span] = []

        current_span = []

        tags = defaultdict(lambda: 0.0)

        if tag_type=='bioes':
            previous_tag_value: str = "O"
            for token in self:
                tag = token.get_tag(tag_type)

                # non-set tags are OUT tags
                if tag_value == "" or tag_value == "O":
                    tag_value = "O-"

                # anything that is not a BIOES tag is a SINGLE tag
                if tag_value[0:2] not in ["B-", "I-", "O-", "E-", "S-"]:
                    tag_value = "S-" + tag_value

                # anything that is not OUT is IN
                in_span = False
                if tag_value[0:2] not in ["O-"]:
                    in_span = True

                # single and begin tags start a new span
                starts_new_span = False
                if tag_value[0:2] in ["B-", "S-"]:
                    starts_new_span = True

                if (
                    previous_tag_value[0:2] in ["S-"]
                    and previous_tag_value[2:] != tag_value[2:]
                    and in_span
                ):
                    starts_new_span = True

                if (starts_new_span or not in_span) and len(current_span) > 0:  # span开头：starts_new_span= True; span结束且tag=O：in_span =False; 
                    spans.append(
                        Span(
                            current_span,
                            tag = previous_tag_value[2:],
                          )
                    )
                    current_span = []

                if in_span:
                    current_span.append(token)

                # remember previous tag
                previous_tag_value = tag_value

            if len(current_span) > 0:
                spans.append(
                    Span(
                        current_span,
                        tag = previous_tag_value[2:],
                    )
                )

            self.spans = spans
        elif tag_type=='srl':
            span_set = set()
            # pdb.set_trace()
            for token in self:
                
                tags_list = token.get_tag(tag_type).split('|')
                if tags_list != ['_']:
                    for tag in tags_list:
                        tag_s = tag.split(':')
                        pred_id = tag_s[0]
                        
                        try:
                            end, label = tag_s[1].split('-',1)
                        except:
                            pdb.set_trace()
                        if ',' in end:
                            end = end.split(',')[1]
                            end = int(end[0:-1])
                        else:
                            end = int(end[1:-1])
                        span_set.add((int(pred_id), label, int(token.idx), int(end)))
            for pred_id, label, token_id, end in span_set:
                tokens = [self.tokens[i] for i in range(token_id-1, end)]
                new_span = Span(tokens=tokens, tag=label, predicate=self.tokens[pred_id-1])
                spans.append(new_span)
                self.add_predicate(int(pred_id))

            self.spans = spans
            self.spans.sort(key=lambda x:(x.predicate_id, x.start_pos), reverse=False)

    def export_spans(self, tag_type: str='ner') -> List:
        # pdb.set_trace()
        if self.spans is None:
            self.get_spans(tag_type)
        ex_list = []
        self.spans.sort(key=lambda x:x.predicate_id)
        if len(self.spans) > 0:
            for span in self.spans:
                ex_list.append(span.export_span())
        return ex_list
    
    def export_texts(self) -> List:
        ex_list = []
        if len(self.tokens) > 0: 
            for token in self.tokens:
                ex_list.append(token.text)
        return ex_list
    

    def to_tokenized_string(self) -> str:
        self.tokenized = " ".join([t.text for t in self.tokens])

        return self.tokenized

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self):
        return 'Sentence: {} "{}" - {} Tokens'.format(self.sent_id, 
            " ".join([t.text for t in self.tokens]), len(self)
        )

    def __str__(self) -> str:

        return f'Sentence: {self.sent_id} "{self.to_tokenized_string()}" - {len(self)} Tokens'

    def __len__(self) -> int:
        return len(self.tokens)


    def export_conll05(self, gold_predicate):
        # pdb.set_trace()
        if len(self.predicates)==0:
            ex_list = ['-'for token in self.tokens]
            ex_line = '\n'.join(ex_list)
        else:
            assert len(self.tokens) == len(gold_predicate)
            ex_list = [['-'] + ['*']*len(self.predicates) for token in self.tokens]  #[[token1, *], [token2, *], ...]
            pred_indices = sorted(list(self.predicates))
            for predicate_id in pred_indices:
                try:
                    predicate = self.tokens[predicate_id-1].text 
                    if gold_predicate[predicate_id-1] != '-':
                        ex_list[predicate_id-1][0] = gold_predicate[predicate_id-1]
                    else:
                        ex_list[predicate_id-1][0] = predicate
                except:
                    pdb.set_trace()

            self.spans.sort(key=lambda x:(x.predicate_id, x.start_pos), reverse=False)
            
            for span in self.spans:
                pred_id = pred_indices.index(span.predicate_id)
                start, end = span.start_pos, span.end_pos
                for i in range(start, end+1):   # token_id = i
                    fill_content = '*'
                    if i==start:
                        fill_content = '(' + span.tag + '*'
                    if i==end:
                        fill_content += ')'
                        
                    ex_list[i-1][pred_id+1] = fill_content

            ex_list = ['\t'.join(line) for line in ex_list]
            ex_line = '\n'.join(ex_list) 
        
        return ex_line + '\n'

    
            


def gen_sentences(file, column_types):
    sent_list = []
    sent_id = None
    lines_in_sent = []  # lines for a sentence
    with open(file, 'r') as f:
        try:
            for line in f:
                line_split = line.split()
                    
                if len(line_split) == 0:
                    if len(lines_in_sent) > 0:
                        # pdb.set_trace()
                        new_sentence = Sentence(sent_id=sent_id, lines=lines_in_sent, column_types=column_types)
                        new_sentence.get_spans(tag_type='srl')
                        sent_list.append(new_sentence)
                        sent_id = None
                        lines_in_sent = []
                elif line_split[0].startswith('#'):
                    sent_id = line_split[0]
                else:
                    lines_in_sent.append(line)
        except:
            pdb.set_trace()
        if len(lines_in_sent) > 0:
            new_sentence = Sentence(sent_id=sent_id, lines=lines_in_sent, column_types=column_types)
            new_sentence.get_spans(tag_type='srl')
            sent_list.append(new_sentence)
    
    return sent_list

    

if __name__=='__main__':

    # import jsonlines


    file = 'a.conllu'
    column_types = ['id', 'form', 'lemma', 'upos', 'xpos', 'pred', 'head', 'deprel', 'srl', 'misc']

    sent_list = gen_sentences(file, column_types)
    ex=sent_list[0].export_conll05()
    pdb.set_trace()