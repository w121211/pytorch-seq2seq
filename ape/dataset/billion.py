import os
from nltk.tokenize.moses import MosesDetokenizer
from mosestokenizer import *

from torchtext.datasets.translation import TranslationDataset


class BillionWord(TranslationDataset):
    path = '/Users/chi/Work/@data/1-billion-word-language-modeling-benchmark-r13output' \
           '/training-monolingual.tokenized.shuffled'
    dirname = '../../data/billion'
    name = 'billion'

    @classmethod
    def corpus_file_paths(cls):
        paths = []
        for root, dirs, files in os.walk(cls.path):
            paths += [os.path.join(root, file) for file in files]
        return paths

    @classmethod
    def preprocess(cls, to_dir='../../data/billion', max_line=None):
        # detokenizer = MosesDetokenizer()

        count = 0
        with MosesDetokenizer('en') as detokenize, \
                open(os.path.join(to_dir, 'train.src'), 'w+') as train_f, \
                open(os.path.join(to_dir, 'val.src'), 'w+') as val_f, \
                open(os.path.join(to_dir, 'test.src'), 'w+') as test_f:
            for root, dirs, files in os.walk(cls.path):
                for file in files:
                    with open(os.path.join(root, file), 'r') as in_f:
                        for line in in_f:
                            # 重建原本的sentence（為了subword tokenizer）
                            # line = detokenize(line.rstrip().split(' '))
                            # line += '\n'

                            if count % 20 == 0:
                                test_f.write(line)
                            elif count % 10 == 0:
                                val_f.write(line)
                            else:
                                train_f.write(line)

                            if max_line is not None and count > max_line:
                                break

                            if count % 100000 == 0:
                                print(count)

                            count += 1

    @classmethod
    def splits(cls, exts, fields, path='./data/billion', root='./data',
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
        return super(BillionWord, cls).splits(
            exts=exts, fields=fields, root=root,
            train=train, validation=validation, test=test, **kwargs)

# BillionWord.preprocess(max_line=1000000)
