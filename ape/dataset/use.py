import os
import re
import spacy
from tqdm import tqdm

from torchtext.datasets.translation import TranslationDataset


class USECorpus(TranslationDataset):
    path = '/Users/chi/Work/@data/Uppsala Student English Corpus (USE)/2457/USEcorpus'
    dirname = '../../data/use'
    name = 'use'

    min_len = 3

    @classmethod
    def preprocess(cls, to_dir='../../data/use'):
        nlp = spacy.load('en_core_web_sm')
        store = {'train': [],
                 'val': [],
                 'test': [], }

        count = 0
        for root, dirs, files in tqdm(os.walk(cls.path)):
            for file in files:
                if not file.endswith('txt'):
                    continue

                with open(os.path.join(root, file), 'r', encoding='latin-1') as f:
                    for line in f:
                        # 跳過 xml tokens, eg: <doc>, <title>
                        if line.startswith('<') and line.endswith('>\n'):
                            continue

                        doc = nlp(line)
                        sents = []
                        for sent in doc.sents:
                            t = sent.text.strip()
                            t = re.sub(r'\(.*\)', '', t)  # 移除括號
                            t = re.sub(r'\[.*\]', '', t)  # 移除[...]
                            t = re.sub(r'\<.*\>', '', t)  # 移除<...>
                            t = nlp(t)

                            if '|' not in t.text and len(t) >= cls.min_len:
                                sents.append(t.text)
                                sents.append(' '.join([token.text for token in sent]))  # tokenize
                        count += len(sents)

                        if count % 20 == 0:
                            store['test'] += sents
                        elif count % 10 == 0:
                            store['val'] += sents
                        else:
                            store['train'] += sents

        for k, v in store.items():
            with open(os.path.join(to_dir, k), 'w+') as f:
                v = map(lambda x: x + '\n', v)
                f.writelines(v)

    @classmethod
    def splits(cls, exts, fields, path='./data/lang8', root='./data',
               train='train', validation='val', test='test', **kwargs):
        return super(USECorpus, cls).splits(
            path, root, train, validation, test, exts=exts, fields=fields, **kwargs)


USECorpus.preprocess()
