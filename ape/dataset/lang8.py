import os
import json
import re
import csv
from random import shuffle

from tqdm import tqdm
from nltk.tokenize.moses import MosesTokenizer
from fuzzywuzzy import fuzz
import spacy

from torchtext import data
from torchtext.datasets.translation import TranslationDataset


class ListDataset(data.Dataset):
    def __init__(self, dataset, fields, **kwargs):
        if not isinstance(data, list):
            raise TypeError(
                'data僅可為list, eg data=[(\'value1 for field1\', \'value2 for field2\')]')

        examples = []
        for d in dataset:
            src_str, tgt_str = d
            if src_str != '' and tgt_str != '':
                examples.append(data.Example.fromlist([src_str, tgt_str], fields))
        super(ListDataset, self).__init__(examples, fields, **kwargs)


class Lang8(TranslationDataset):
    path = '/Users/chi/Work/@data/lang-8/lang-8-20111007-2.0/lang-8-20111007-L1-v2.dat'
    # path = '/home/chi/Documents/ape/data/lang-8/lang-8-20111007-2.0/lang-8-20111007-L1-v2.dat'
    dirname = '../../data/lang8'
    name = 'lang8'

    LABEL_ERR = 0  # label for error sentence
    LABEL_COR = 1  # label for corrected sentence

    @classmethod
    def preprocess(cls, to_dir='../../data/lang8', min_len=3):
        ''' 建立lang-8的train & test datasets '''

        nlp = spacy.load('en_core_web_lg')

        store = {
            'train': [],
            'val': [],
            'test': [], }

        count = 0
        is_ascii = lambda s: len(s) == len(s.encode())
        with open(cls.path) as f:
            for line in f:
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if j[2] != 'English':
                    continue

                for i in range(len(j[4])):
                    err = j[4][i]
                    err = nlp(err)

                    # 簡單測試是否為句子 & 是否包含非英文的內容
                    if any([not err.text.endswith(('.', '?', '!')),
                            not is_ascii(err.text),
                            len(err) < min_len]):
                        continue

                    # 有數個更改者，不一定每個都有更改
                    for cor in j[5][i]:
                        if cor is None or not is_ascii(cor):
                            continue

                        # remove tags [f-blue], [f-red], [f-bold]
                        for x in ('[f-blue]', '[f-red]', '[f-bold]'):
                            cor = cor.replace(x, '').replace(x.replace('[', '[/'), '')
                        cor = re.sub(r'\[sline\].*\[/sline\]', '', cor)  # remove [sline] ... [/sline]
                        cor = re.sub(r'\(.*\)', ' ', cor)  # 移除括號
                        cor = re.sub(r'\[.*\]', ' ', cor)  # 移除中括號
                        cor = re.sub(r'\<.*\>', ' ', cor)  # 移除中括號

                        cor = nlp(cor)
                        if any([len(cor) < min_len,
                                err.text == cor.text,
                                err.similarity(cor) < 0.8]):
                            continue

                        # export
                        pair = (' '.join([tk.text.strip() for tk in err]),
                                ' '.join([tk.text.strip() for tk in cor]))
                        if count % 40 == 0:
                            store['val'].append(pair)
                        elif count % 20 == 0:
                            store['test'].append(pair)
                        else:
                            store['train'].append(pair)

                count += 1
                # if count % 100 == 0:
                #     break
                if count % 10000 == 0:
                    print(count)

        def write_to(paths, data_pairs):
            with open(paths[0], 'w+') as src_f, open(paths[1], 'w+') as tgt_f:
                for src, tgt in data_pairs:
                    src_f.write(src + '\n')
                    tgt_f.write(tgt + '\n')

        for k, v in store.items():
            write_to((os.path.join(to_dir, '%s.err' % k),
                      os.path.join(to_dir, '%s.cor' % k)), v)

    @classmethod
    def preprocess_gan(cls, to_dir='../../data/lang8', ratio=10):
        if not os.path.exists(os.path.join(to_dir, 'train.err')):
            cls.preprocess(to_dir)

        def split(path):
            if '.err' in path:
                pretrain_path = path.replace('.err', '.pre.err')
                advtrain_path = path.replace('.err', '.adv.err')
            else:
                pretrain_path = path.replace('.cor', '.pre.cor')
                advtrain_path = path.replace('.cor', '.adv.cor')
            with open(path, 'r') as inp, open(pretrain_path, 'w+') as pre, open(advtrain_path, 'w+') as adv:
                lines = [line for line in inp]
                pre_lines = int(len(lines) / 20)
                for i in range(0, pre_lines):
                    pre.write(lines[i])
                for i in range(pre_lines, len(lines)):
                    adv.write(lines[i])

        for path in [os.path.join(to_dir, f_name) for f_name in ('train.err', 'train.cor', 'test.err', 'test.cor')]:
            split(path)

    @classmethod
    def preprocess_label(cls):
        # train
        for prefix in ['train', 'test']:
            with open(os.path.join(cls.dir, prefix + '.cor')) as train_cor, \
                    open(os.path.join(cls.dir, prefix + '.err')) as train_err, \
                    open(os.path.join(cls.dir, prefix + '.label.tsv'), 'w+') as tsv:
                outputs = []  # (sentences, label) pair
                for line in train_err:
                    outputs.append((line, cls.LABEL_ERR))
                for line in train_cor:
                    outputs.append((line, cls.LABEL_COR))
                shuffle(outputs)

                writer = csv.DictWriter(tsv, delimiter='\t', fieldnames=['sentence', 'label'])
                writer.writeheader()
                for row in outputs:
                    writer.writerow({'sentence': row[0], 'label': row[1]})

    @classmethod
    def splits(cls, exts, fields, root='./data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of the Lang8 dataset.
        Arguments:
            root: Root dataset storage directory. Default is '.data'.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.

        Usage:
        >>> src = SourceField()
        >>> tgt = TargetField()
        >>> train, dev, test = Lang8.splits(
            exts=('.err', '.cor'), fields=[('src', src), ('tgt', tgt)],
            train='test', validation='test', test='test')
        """
        return super(Lang8, cls).splits(exts=exts, fields=fields, root=root,
                                        train=train, validation=validation, test=test, **kwargs)


# Lang8.preprocess()
