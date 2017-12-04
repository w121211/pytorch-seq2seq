import logging
import os
import sys

import torch
import torchtext
from torch.optim.lr_scheduler import StepLR

seq2seq_pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if seq2seq_pardir not in sys.path:
    sys.path.insert(0, seq2seq_pardir)

from seq2seq.trainer import SupervisedTrainer, trainer
from seq2seq.trainer import gan
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.models.cnn import ClassifierCNN
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.dataset.lang8 import Lang8
from seq2seq.util import helper

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)

# Params
random_seed = 80
checkpoint = ''
resume = False

max_len = 50
min_len = 5

hidden_size = 6  # encoder/decoder hidden size
bidirectional = True


def pretrain_generator(model, train, dev):
    # pre-train generator
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    optimizer = Optimizer(torch.optim.Adam(gen.parameters()), max_grad_norm=5)
    scheduler = StepLR(optimizer.optimizer, 1)
    optimizer.set_scheduler(scheduler)

    supervised = SupervisedTrainer(loss=loss, batch_size=32, random_seed=random_seed)
    model = supervised.train(model, train, num_epochs=50, dev_data=dev,
                             optimizer=optimizer, teacher_forcing_ratio=0, resume=resume)


def len_filter(example):
    return min_len <= len(example.src) <= max_len and min_len <= len(example.tgt) <= max_len


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
# real_iter = torchtext.data.BucketIterator.splits(
#     adv_train, batch_sizes=256, device=device,
#     sort_key=lambda x: len(x.src))
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

# pre-train generator
# pretrain_generator(gen, pre_train, pre_dev)

# init discriminator
dis = ClassifierCNN(len(tgt.vocab), embed_dim=hidden_size,
                    num_class=1, num_kernel=100,
                    kernel_sizes=[3, 4, 5], dropout_p=0.25)
if torch.cuda.is_available():
    dis.cuda()

# init trainers
gen_trainer = gan.PolicyGradientTrainer(max_len=max_len)
gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.01)

dis_trainer = trainer.BinaryClassifierTrainer(print_every=50)
dis_optimizer = torch.optim.Adam(dis.parameters(), lr=0.01)

# pre-train discriminator
samples = [sample for sample, _, _, _ in gen_trainer.gen_sample(
    gen, adv_train_iter, num_src=20, src2sample=1)]
batch = next(iter(real_iter))
reals = batch.tgt.data[:, 1:]  # 裁掉<sos>
_train_iter = helper.batch_gen(samples, reals, pad_id, max_len, batch_size=16)
dis_trainer.train(dis, _train_iter, dis_optimizer, dev_iter=adv_dev_iter)

# adversarial training
logging.info('Start adversarial training')
# for epoch in range(1, 20):
#     logging.info('[Epoch %d]: start to train generator' % epoch)
#
#     # train generator
#     gen_trainer.train(gen, dis, adv_train_iter, dev_data=adv_dev, optimizer=gen_optimizer)
#
#     # train discriminator
#     logging.info('[Epoch %d]: start to train discriminator' % epoch)
#
#     samples = [sample for sample, _, _, _ in gen_trainer.gen_sample(
#         gen, adv_train_iter, num_src=256, src2sample=1)]
#     batch = next(iter(real_iter))
#     reals = batch.tgt.data[:, 1:]  # 裁掉<sos>
#     dis_trainer.train(dis, helper.gen_batch(
#         samples, reals, pad_id, max_len, batch_size=3), dis_optimizer)

# logging.info('[Epoch %d]: dis loss %.6f' % (epoch, loss.data[0]))
