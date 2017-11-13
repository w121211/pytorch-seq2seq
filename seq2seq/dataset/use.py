import os

import nltk.data
from nltk.tokenize.moses import MosesTokenizer

from torchtext.datasets.translation import TranslationDataset


class USE(TranslationDataset):
    path = '/Users/chi/Work/@data/Uppsala Student English Corpus (USE)/2457/USEcorpus'
    dirname = '../../data/use'
    name = 'Uppsala_Student_English_Corpus'

    @classmethod
    def preprocess(cls, to_dir='../../data/use'):
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        tokenizer = MosesTokenizer()

        train_path = os.path.join(to_dir, 'train.src')
        test_path = os.path.join(to_dir, 'test.src')

        count = 0
        with open(train_path, 'w+') as train, open(test_path, 'w+') as test:
            for root, dirs, files in os.walk(cls.path):
                for file in files:
                    if not file.endswith('txt'):
                        continue

                    with open(os.path.join(root, file), 'r', encoding='latin-1') as f:
                        for line in f:
                            # 跳過 xml tokens, eg: <doc>, <title>
                            if line.startswith('<') and line.endswith('>\n'):
                                continue

                            sents = sent_detector.tokenize(line.strip())
                            sents = [' '.join(tokenizer.tokenize(sent)) for sent in sents]

                            if count % 10 == 0:
                                test.write('\n'.join(sents))
                            else:
                                train.write('\n'.join(sents))
                            if count % 1000 == 0:
                                print(count)
                            count += 1

    @classmethod
    def splits(cls, exts, fields, path='./data/lang8', root='./data',
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
        """
        return super(USE, cls).splits(
            path, root, train, validation, test, exts=exts, fields=fields, **kwargs)


USE.preprocess()
