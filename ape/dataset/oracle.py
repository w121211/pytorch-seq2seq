import os
import json
import re
import csv
from random import shuffle
import xml.etree.ElementTree as ET

import torch

from torchtext.datasets.translation import TranslationDataset


class Oracle(TranslationDataset):
    path = '/Users/chi/Work/@data/oracle/oracle_samples.trc'
    dirname = '../../data/oracle'
    name = 'Oracle'

    @classmethod
    def preprocess(cls, to_dir='../../data/oracle'):
        torch.load(cls.path)

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
