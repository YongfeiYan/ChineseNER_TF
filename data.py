import os
import sys
import jieba
from torchtext.data import Example, Dataset
from itertools import chain
import re
from collections import Counter
from seqeval.metrics import classification_report
import numpy as np

from tfutils.preprocessing import Vocab, Field, BucketIterator
from tfutils.utils import LoaderWrapper


PAD = "<PAD>"  # padding
SOS = "<SOS>"  # start of sequence
EOS = "<EOS>"  # end of sequence
UNK = "<UNK>"  # unknown token


def accuracy_report(pred_path, gold_true_path):
    """Evaluation f1 score.
    Data format in pred_path and gold_true_path(each line contains a char and its tag, \t separator. \n separates
    different sentences):
        char tag
        char tag
        \n
        char tag
        char tag
        ...
    """
    y_pred = []
    y_true = []
    per = [0] * 3
    org = [0] * 3
    loc = [0] * 3
    avg = [0] * 3

    with open(pred_path, encoding='utf-8') as f1:
        for line in f1:
            if line.strip():
                y_pred.append(line.split()[-1])
    with open(gold_true_path, encoding='utf-8') as f2:
        for line in f2:
            if line.strip():
                y_true.append(line.split()[-1])

    if len(y_true) != len(y_pred):
        print("Length of your prediction should be equal to gold's.")
    else:
        report = classification_report(y_true, y_pred, 4)
        print(report)

        def _parse(line):
            return [float(s) for s in line.split()[-4:-1]]

        lines = report.split('\n')
        per = _parse(lines[2])
        org = _parse(lines[3])
        loc = _parse(lines[4])
        avg = _parse(lines[6])
    return {'p': avg[0], 'r': avg[1], 'f1': avg[2]}


def read_lines(path):
    """
    return [ [list of words, list of tags]... ]
    """
    def _add(sent_tag, lst):
        if len(sent_tag[0]) > 0:
            lst.append(sent_tag)
    data = []
    with open(path) as f:
        sentence = [[], []]
        for line in f:
            if line == '\n':
                _add(sentence, data)
                sentence = [[], []]
            else:
                try:
                    line = line.rstrip("\n")
                    token = re.findall('\s', line)[-1]
                    last_idx = line.rfind(token)
                    w = line[:last_idx]
                    t = line[last_idx+1:]
                    sentence[0].append(w)
                    sentence[1].append(t)

                except :
                    print('|' + line + '|')
                    raise RuntimeError("File format error")
        _add(sentence, data)
    return data


def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append('S')
        else:
            tmp = ['I'] * len(word)
            tmp[0] = 'B'
            tmp[-1] = 'E'
            seg_feature.extend(tmp)
    return seg_feature


def to_dict(batch):
    return {'sent': batch.sent[0],
            'tag': batch.tag,
            'seg': batch.seg
            }


def get_iterator(data_path, batch_size=32, shuffle=True, vocab_size=10000, fields=None):
    """If fields is None, then this is training data which will be used to build vocabulary."""
    data = read_lines(data_path)
    data = [[s, t, get_seg_features(''.join(s))] for s, t in data]
    if fields is None:
        f_sent = Field(sequential=True, use_vocab=True, include_lengths=True, batch_first=True,
                                 pad_token=PAD, unk_token=UNK, dtype=np.int64)
        f_sent.vocab = Vocab(Counter(chain(*[d[0] for d in data])),
                             max_size=vocab_size, specials=[PAD, UNK], unk_idx=1)
        v_tag = Vocab(Counter(chain(*[d[1] for d in data])), max_size=100000)  # use all tags and padding

        f_tag = Field(sequential=True, use_vocab=True, batch_first=True, pad_token=v_tag.itos[0], dtype=np.int64)
        f_tag.vocab = v_tag
        v_seg = Vocab(Counter(chain(*[d[2] for d in data])), max_size=10000000)

        f_seg = Field(sequential=True, use_vocab=True, batch_first=True, pad_token=v_seg.itos[0], dtype=np.int64)
        f_seg.vocab = v_seg
        fields = [('sent', f_sent), ('tag', f_tag), ('seg', f_seg)]
    examples = [Example.fromlist(d, fields) for d in data]
    dataset = Dataset(examples, fields)
    loader = BucketIterator(dataset, batch_size=batch_size, shuffle=shuffle, train=shuffle, sort=shuffle,
                            sort_within_batch=False, sort_key=lambda x: len(x.sent))

    return LoaderWrapper(loader, transform=to_dict), fields


if __name__ == '__main__':

    print('finished')
