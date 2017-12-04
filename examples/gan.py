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
from seq2seq.evaluator import Evaluator
from seq2seq.dataset import SourceField, TargetField
from seq2seq.dataset.lang8 import Lang8
from seq2seq.util import helper, checkpoint, logger

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(filename='exp.log', format=LOG_FORMAT, level=logging.DEBUG)

# logger = logger.Logger('./logs')

# Params
random_seed = 80
expt_gen_dir = './experiment/gen/'
expt_dis_dir = './experiment/dis/'
resume = False

max_len = 50
min_len = 5

hidden_size = 256  # encoder/decoder hidden size
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

    supervised = SupervisedTrainer(loss=loss, batch_size=32, random_seed=random_seed, expt_dir=expt_gen_dir)
    supervised.train(model, train, num_epochs=20, dev_data=dev,
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
logging.info('Pre-training generator...')
pretrain_generator(gen, pre_train, pre_dev)

# init discriminator
dis = ClassifierCNN(len(tgt.vocab), embed_dim=hidden_size,
                    num_class=1, num_kernel=100, kernel_sizes=[3, 4, 5], dropout_p=0.25)
if torch.cuda.is_available():
    dis.cuda()

# init trainers, optimizers, evaluators
g_trainer = gan.PolicyGradientTrainer(max_len=max_len, logger=logger)
g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.01)
g_evaluator = Evaluator()

d_trainer = trainer.BinaryClassifierTrainer()
d_optimizer = torch.optim.Adam(dis.parameters(), lr=0.01)

# pre-train discriminator
samples = [sample for sample, _, _, _ in g_trainer.gen_sample(
    gen, adv_train_iter, num_src=256, src2sample=1)]
batch = next(iter(real_iter))
reals = batch.tgt.data[:, 1:]  # 裁掉<sos>
for epoch in range(1, 20 + 1):
    logging.info('Epoch[%d]' % epoch)
    _train_iter = helper.batch_gen(samples, reals, pad_id, max_len, batch_size=16)
    d_trainer.train_epoch(dis, _train_iter, d_optimizer, dev_iter=adv_dev_iter)

# adversarial training
logging.info('Start adversarial training')

g_step, d_step = 0, 0
for epoch in range(1, 20):
    logging.info('[Epoch %d]: train generator' % epoch)

    # train generator
    g_step = g_trainer.train(gen, dis, adv_train_iter, dev_data=adv_dev, optimizer=g_optimizer, step=g_step)

    # evalutate generator
    dev_loss, accuracy = g_evaluator.evaluate(gen, adv_dev)
    logging.info('Dev %s: %.4f, Accuracy: %.4f' % ('gen NLLloss', dev_loss, accuracy))
    logger.scalar_summary('G-NLLloss', dev_loss, epoch)

    # train discriminator
    logging.info('[Epoch %d]: train discriminator' % epoch)

    samples = [sample for sample, _, _, _ in g_trainer.gen_sample(
        gen, adv_train_iter, num_src=256, src2sample=1)]
    batch = next(iter(real_iter))
    reals = batch.tgt.data[:, 1:]  # 裁掉<sos>
    for _epoch in range(1, 20 + 1):
        _train_iter = helper.batch_gen(samples, reals, pad_id, max_len, batch_size=16)
        d_step = d_trainer.train_epoch(dis, _train_iter, d_optimizer, dev_iter=adv_dev_iter, step=d_step)

    # evalutate discriminator
    samples = [sample for sample, _, _, _ in g_trainer.gen_sample(
        gen, adv_train_iter, num_src=256, src2sample=1)]
    batch = next(iter(real_iter))
    reals = batch.tgt.data[:, 1:]  # 裁掉<sos>
    _train_iter = helper.batch_gen(samples, reals, pad_id, max_len, batch_size=16)
    loss, accuracy = d_trainer.evaluate(dis, _train_iter)
    logger.scalar_summary('D-CrossEntropyLoss', loss, epoch)
    logger.scalar_summary('D-Accuracy', accuracy, epoch)

    # save models
    checkpoint.Checkpoint(model=gen, optimizer=g_optimizer, epoch=epoch, step=0,
                          input_vocab=src.vocab, output_vocab=tgt.vocab).save(expt_gen_dir)
    checkpoint.Checkpoint(model=dis, optimizer=d_optimizer, epoch=epoch, step=0,
                          input_vocab=src.vocab, output_vocab=tgt.vocab).save(expt_dis_dir)
