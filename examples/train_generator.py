import logging
import os
import sys

import torch
import torchtext

seq2seq_pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if seq2seq_pardir not in sys.path:
    sys.path.insert(0, seq2seq_pardir)

from seq2seq.trainer import gan
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.dataset import SourceField, TargetField
from seq2seq.dataset.lang8 import Lang8

# from seq2seq.logger import Logger

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)

# logger = Logger('./logs')  # 用於tensorboard

# Params
random_seed = 80
checkpoint = ''
resume = True

max_len = 50
min_len = 5

hidden_size = 256  # encoder/decoder hidden size
bidirectional = True

# prepare dataset
src = SourceField()
tgt = TargetField()
device = None if torch.cuda.is_available() else -1
pre_train, pre_dev, pre_test = Lang8.splits(
    exts=('.pre.cor', '.pre.err'), fields=[('src', src), ('tgt', tgt)],
    train='test', validation='test', test='test')
adv_train, adv_dev, adv_test = Lang8.splits(
    exts=('.adv.cor', '.adv.err'), fields=[('src', src), ('tgt', tgt)],
    train='test', validation='test', test='test')
adv_train_iter, adv_dev_iter, real_iter = torchtext.data.BucketIterator.splits(
    (adv_train, adv_dev, adv_train), batch_sizes=(1, 256, 256), device=device,
    sort_key=lambda x: len(x.src))
src.build_vocab(pre_train, pre_dev, pre_test, adv_train, adv_dev, adv_test)
tgt.build_vocab(pre_train, pre_dev, pre_test, adv_train, adv_dev, adv_test)
pad_id = tgt.vocab.stoi[tgt.pad_token]

# init generator
encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                     bidirectional=bidirectional, rnn_cell='lstm', variable_lengths=True)
decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                     dropout_p=0.2, use_attention=True, bidirectional=bidirectional, rnn_cell='lstm',
                     eos_id=tgt.eos_id, sos_id=tgt.sos_id)
gen = Seq2seq(encoder, decoder)
if torch.cuda.is_available():
    logging.info('Use Cuda')
    gen.cuda()
for param in gen.parameters():
    param.data.uniform_(-0.08, 0.08)

t = gan.PolicyGradientTrainer(max_len=max_len)
optimizer = torch.optim.Adam(gen.parameters(), lr=0.01)
samples = [sample for sample, _, _, _ in t.gen_sample(
    gen, adv_train_iter, num_src=100, src2sample=13)]
