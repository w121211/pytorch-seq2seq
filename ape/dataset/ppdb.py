import os
import re
import spacy
from tqdm import tqdm

from torchtext.datasets.translation import TranslationDataset


class PPDB(TranslationDataset):
    path = '/Users/chi/Work/@data/ppdb-2.0-s-all'
    dirname = '../../data/ppdb'
    name = 'ppdb'

    min_len = 3

    @classmethod
    def preprocess(cls, to_dir='../../data/ppdb'):
        store = {'corpus.a': [], 'corpus.b': []}

        with open(cls.path, 'r') as f:
            for line in tqdm(f):
                def clean(s):
                    s = s.strip()
                    s = re.sub(r'\(.*\)', '', s)  # 移除括號
                    s = re.sub(r'\[.*\]', '', s)  # 移除[...]
                    s = re.sub(r'\<.*\>', '', s)  # 移除<...>
                    return s

                d = line.split('|||')
                d = (clean(d[1]), clean(d[2]))
                if not (len(d[0]) == 0 or len(d[1]) == 0):
                    store['corpus.a'].append(d[0])
                    store['corpus.b'].append(d[1])

        for k, v in store.items():
            with open(os.path.join(to_dir, k), 'w+') as f:
                f.writelines(map(lambda x: x + '\n', v))

    @classmethod
    def splits(cls, exts, fields, path='./data/lang8', root='./data',
               train='train', validation='val', test='test', **kwargs):
        return super(USECorpus, cls).splits(
            path, root, train, validation, test, exts=exts, fields=fields, **kwargs)


PPDB.preprocess()
