import os
import sys
import inspect
import argparse
import logging
import random

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchtext

seq2seq_pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if seq2seq_pardir not in sys.path:
    sys.path.insert(0, seq2seq_pardir)

import seq2seq
from seq2seq.trainer import trainer
from seq2seq.trainer import reinforce
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.models.classifierCNN import ClassifierCNN
from seq2seq.dataset import SourceField, TargetField
from seq2seq.dataset.lang8 import Lang8

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)


# prepare data
# logging.info('Prepare data')
# device = None if torch.cuda.is_available() else -1
# train, dev, test = Lang8.splits(
#     exts=('.err', '.cor'), fields=[('src', src), ('tgt', tgt)],
#     train='test', validation='test', test='test')
# train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(
#     (train, dev, test), batch_sizes=(16, 256, 256), device=device,
#     sort=True, sort_key=lambda x: len(x.src), )
# src_field.build_vocab(train, dev, test)
# tgt_field.build_vocab(train, dev, test)
# src_field.build_vocab(train)
# tgt_field.build_vocab(train)


class SimpleBatch(object):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt


def gen_batch(samples, reals, pad_id, max_len, batch_size=16):
    '''
    非常naive的batch generator，用於將samples與reals的tensors合併、shuffle、然後產生batches。

    argument:
        samples list of Tensors
        reals  a Tensor (batch, seq_len)

    return:
        Variable (batch, max_seq_len) naive的將所有不足長度的seq全部pad成max_len
    '''
    reals = [(src, src.size(1), 1) for src in reals.chunk(reals.size(0))]  # (Tensor, seq_len, tgt_label)
    samples = [(src, src.size(1), 0) for src in samples]

    examples = reals + samples
    random.shuffle(examples)
    for i in range(0, len(examples), batch_size):
        src = autograd.Variable(torch.LongTensor(batch_size, max_len).fill_(pad_id))
        tgt = autograd.Variable(torch.zeros(batch_size, 1).float())

        for j, (_seq, seq_len, label) in enumerate(examples[i:i + batch_size]):
            src[j, :seq_len] = _seq
            tgt[j, 0] = label
        yield SimpleBatch(src, tgt)


max_len = 50
min_len = 5


def len_filter(example):
    return min_len <= len(example.src) <= max_len and \
           min_len <= len(example.tgt) <= max_len


src = SourceField()
tgt = TargetField()
device = None if torch.cuda.is_available() else -1
train = torchtext.data.TabularDataset(
    path='data/toy_ascend/train/data.txt', format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter)
dev = torchtext.data.TabularDataset(
    path='data/toy_ascend/dev/data.txt', format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter)
train_iter = torchtext.data.BucketIterator(
    train, batch_size=1,
    sort_key=lambda x: len(x.src), device=device)
real_iter = torchtext.data.BucketIterator(
    train, batch_size=200,
    sort_key=lambda x: len(x.src), device=device)
src.build_vocab(train)
tgt.build_vocab(train)
pad_id = tgt.vocab.stoi[tgt.pad_token]

# init generator
hidden_size = 6
bidirectional = True
encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                     bidirectional=bidirectional, rnn_cell='lstm', variable_lengths=True)
decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                     dropout_p=0.2, use_attention=True, bidirectional=bidirectional, rnn_cell='lstm',
                     eos_id=tgt.eos_id, sos_id=tgt.sos_id)
gen = Seq2seq(encoder, decoder)
if torch.cuda.is_available():
    gen.cuda()
for param in gen.parameters():
    param.data.uniform_(-0.08, 0.08)

# init discriminator
dis = ClassifierCNN(len(tgt.vocab), embed_dim=hidden_size,
                    num_class=1, num_kernel=100,
                    kernel_sizes=[3, 4, 5], dropout_p=0.2)
if torch.cuda.is_available():
    dis.cuda()

# init trainers
gen_trainer = reinforce.PolicyGradientTrainer(max_len=max_len)
gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.01)

dis_trainer = trainer.ClassifierTrainer()
dis_optimizer = torch.optim.Adam(dis.parameters(), lr=0.01)

# pre-train discriminator
samples = [sample for sample, _, _, _ in gen_trainer.gen_sample(
    gen, train_iter, num_src=10, src2sample=1)]
batch = next(iter(real_iter))
reals = batch.tgt.data[:, 1:]  # 裁掉<sos>
dis_trainer.train(dis, gen_batch(samples, reals, pad_id, max_len, batch_size=3),
                  dis_optimizer)

# adversarial training
logging.info('Start adversarial training')
for epoch in range(1, 20):
    logging.info('[Epoch %d]: start to train generator' % epoch)
    # train generator
    loss = gen_trainer.train(gen, dis, train_iter, dev_data=dev,
                             optimizer=gen_optimizer)
    # logging.info('[Epoch %d]: gen loss %.6f' % (epoch, loss.data[0]))

    # train discriminator
    logging.info('[Epoch %d]: train discriminator' % epoch)
    samples = [sample for sample, _, _, _ in gen_trainer.gen_sample(
        gen, train_iter, num_src=10, src2sample=1)]
    batch = next(iter(real_iter))
    reals = batch.tgt.data[:, 1:]
    loss = dis_trainer.train(dis, gen_batch(samples, reals, pad_id, max_len, batch_size=3),
                             dis_optimizer)
    logging.info('[Epoch %d]: dis loss %.6f' % (epoch, loss.data[0]))

# src, tgt = next(iter(gen_batch(samples, reals, pad_id, max_len, batch_size=3)))
# print(src)
# print(tgt)


# t = trainer.ClassifierTrainer()
# t.train(dis, train_iter, dev_iter, optimizer, num_epochs=10)

# weight = torch.ones(len(tgt_field.vocab))
# pad = tgt_field.vocab.stoi[tgt_field.pad_token]
# loss = Perplexity(weight, pad)
# if torch.cuda.is_available():
#     loss.cuda()

# train
# logging.info('Start training')
# expt_dir = './experiment'
# optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
# optimizer = torch.optim.Adam(seq2seq.parameters())
# optimizer = None

# trainer = PolicyGradientTrainer(
#     num_sample=3, max_len=max_len)
# trainer.train(seq2seq, batch_iter, optimizer=optimizer)


# t = SupervisedTrainer(loss=loss, batch_size=32,
#                       checkpoint_every=100,
#                       print_every=1, expt_dir=expt_dir)
# seq2seq = t.train(seq2seq, train,
#                   num_epochs=10, dev_data=dev,
#                   optimizer=optimizer,
#                   teacher_forcing_ratio=0.5,
#                   resume=False)
