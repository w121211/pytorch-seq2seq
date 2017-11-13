import os
import sys
import random
import logging

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data
from torchtext import datasets

seq2seq_pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if seq2seq_pardir not in sys.path:
    sys.path.insert(0, seq2seq_pardir)

from seq2seq.models.classifierCNN import ClassifierCNN
from seq2seq.loss import NLLLoss
from seq2seq.evaluator import Evaluator




def main():
    # params
    min_len = 5
    batch_size = 10
    embed_dim = 128
    num_kernal = 100
    kernal_sizes = [3, 4, 5]
    dropout_p = 0.5

    lr = 0.01  # initial learning rate

    # load data
    print("\nLoading data...")
    text_field = data.Field(lower=True, batch_first=True)  # text
    label_field = data.Field(sequential=False)  # label

    def len_filter(example):
        # print(example.text)
        return len(example.text) >= min_len

    train, dev, test = datasets.SST.splits(
        text_field, label_field, fine_grained=True, filter_pred=len_filter)
    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)

    device = None if torch.cuda.is_available() else -1
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_sizes=(batch_size, len(dev), len(test)),
        device=device, repeat=False)

    # update args and print
    num_vocab = len(text_field.vocab)
    num_class = len(label_field.vocab) - 1  # vocab size=6，但實際class數量=5

    # args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    # args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #
    # print("\nParameters:")
    # for attr, value in sorted(args.__dict__.items()):
    #     print("\t{}={}".format(attr.upper(), value))

    # new model or load
    cnn = ClassifierCNN(
        num_vocab, embed_dim, num_class, num_kernal, kernal_sizes, dropout_p)

    if torch.cuda.is_available():
        cnn.cuda()

    # train
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    t = Trainer(loss=nn.CrossEntropyLoss())
    t.train(cnn, train_iter, dev_iter, optimizer, num_epochs=10)

    # evaluate
    t.evaluate(cnn, test_iter)

    # if args.predict is not None:
    #     label = train.predict(args.predict, cnn, text_field, label_field)
    #     print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
    # elif args.test:
    #     try:
    #         train.eval(test_iter, cnn, args)
    #     except Exception as e:
    #         print("\nSorry. The test dataset doesn't  exist.\n")
    # else:
    #     print()
    #     train.train(train_iter, dev_iter, cnn, args)


if __name__ == '__main__':
    main()
