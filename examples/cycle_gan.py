import logging
import os
import sys
import re

import torch
from torchtext import data, datasets
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import spacy

seq2seq_pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if seq2seq_pardir not in sys.path:
    sys.path.insert(0, seq2seq_pardir)

import seq2seq
from seq2seq.trainer.gan import CycleGanReinforceTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, gan
from seq2seq.models.transformer import Constants
from seq2seq.dataset.lang8 import Lang8
from seq2seq.util.checkpoint import Checkpoint

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(filename='exp.log', format=LOG_FORMAT, level=logging.DEBUG)
# logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

# spacy_de = spacy.load('de')
# spacy_en = spacy.load('en')

url = re.compile('(<url>.*</url>)')

# Params
random_seed = 80
resume_G, resume_D = False, False

max_len = 50
min_len = 3

hidden_size = 256  # encoder/decoder hidden size
n_layers = 1
bidirectional = True


def len_filter(example):
    return min_len <= len(example.real_a) <= max_len and min_len <= len(example.real_b) <= max_len


# prepare dataset
logging.info('Prepare dataset...')
device = None if torch.cuda.is_available() else -1

BD = data.ReversibleField(init_token='<sos>', eos_token='<eos>',
                          batch_first=True, include_lengths=True, fix_length=max_len + 2)
GD = data.ReversibleField(init_token='<sos>', eos_token='<eos>',
                          batch_first=True, include_lengths=True, fix_length=max_len + 2)
train, val = Lang8.splits(
    exts=('.err', '.cor'), fields=[('real_a', BD), ('real_b', GD)],
    # validation='test', test=None,
    train='test.pre', validation='test.pre', test=None,
    filter_pred=len_filter)
# adv_train, adv_dev, adv_test = Lang8.splits(
#     exts=('.adv.cor', '.adv.err'), fields=[('src', src), ('tgt', tgt)],
#     train='test', validation='test', test='test')
BD.build_vocab(train)
GD.build_vocab(train)

assert BD.vocab.stoi[BD.init_token] == Constants.BOS
assert BD.vocab.stoi[BD.eos_token] == Constants.EOS
assert BD.vocab.stoi[BD.pad_token] == Constants.PAD
assert BD.vocab.stoi[BD.unk_token] == Constants.UNK

# init generators & discriminators
encoder_a = EncoderRNN(len(BD.vocab), max_len, hidden_size, n_layers=n_layers,
                       bidirectional=bidirectional, rnn_cell='lstm', variable_lengths=False)
decoder_a = DecoderRNN(len(GD.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                       n_layers=n_layers, dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                       rnn_cell='lstm', sos_id=GD.vocab.stoi[GD.init_token], eos_id=GD.vocab.stoi[GD.eos_token])
encoder_b = EncoderRNN(len(GD.vocab), max_len, hidden_size, n_layers=n_layers,
                       bidirectional=bidirectional, rnn_cell='lstm', variable_lengths=False)
decoder_b = DecoderRNN(len(BD.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                       n_layers=n_layers, dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                       rnn_cell='lstm', sos_id=BD.vocab.stoi[BD.init_token], eos_id=BD.vocab.stoi[BD.eos_token])
model = gan.CycleGAN(
    g_a=gan.ReinforceGenerator(encoder_a, decoder_a),
    g_b=gan.ReinforceGenerator(encoder_b, decoder_b),
    d_a=gan.Discriminator(len(BD.vocab), embed_dim=hidden_size, num_kernel=100,
                          kernel_sizes=[2, 3, 4, 5], dropout_p=0.2),
    d_b=gan.Discriminator(len(GD.vocab), embed_dim=hidden_size, num_kernel=100,
                          kernel_sizes=[2, 3, 4, 5], dropout_p=0.2))
if torch.cuda.is_available():
    logging.info('Use CUDA')
    model.cuda()

trainer = CycleGanReinforceTrainer(model, A_FIELD=BD, B_FIELD=GD, print_every=6400,
                                   src_field_name='real_a', tgt_field_name='real_b')

# pre-train G
logging.info('Pre-training G...')

if resume_G:
    checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint('./experiment/cycle-seqgan-lstm/pretrained'))
    model = checkpoint.model
else:
    seq2seq.src_field_name = 'real_a'
    seq2seq.tgt_field_name = 'real_b'
    trainer.pretrain_G(trainer.model.g_a, train, val=val,
                       criterion=trainer.criterion_G, optimizer=trainer.optimizer_G,
                       num_epoch=20, FIELD_TGT=GD)

    # seq2seq.src_field_name = 'real_b'
    # seq2seq.tgt_field_name = 'real_a'
    # trainer.pretrain_G(trainer.model.g_b, pre_train, val=pre_val,
    #                    criterion=trainer.criterion_G, optimizer=trainer.optimizer_G,
    #                    num_epoch=20, FIELD_TGT=DE)

    # Checkpoint(model=model, optimizer=None, epoch=0, step=0,
    #            input_vocab=None, output_vocab=None).save('./experiment/pretrained')

# pre-train D
# logging.info('Pre-training D...')
#
# if resume_D:
#     checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint('./experiment/pretrained'))
#     model = checkpoint.model
# else:
#     trainer.pretrain_D(trainer.model.d_a, trainer.model.g_b, pre_train,
#                        criterion=trainer.criterion_D, optimizer=trainer.optimizer_D_a, num_epoch=1,
#                        src_field_name='real_b', tgt_field_name='real_a')
#     trainer.pretrain_D(trainer.model.d_b, trainer.model.g_a, pre_train,
#                        criterion=trainer.criterion_D, optimizer=trainer.optimizer_D_b, num_epoch=1,
#                        src_field_name='real_a', tgt_field_name='real_b')
#     Checkpoint(model=model, optimizer=None, epoch=0, step=0,
#                input_vocab=None, output_vocab=None).save('./experiment/pretrained')
#
# # adversarial training
# logging.info('Start CycleGAN training...')
# trainer.train(gan_train, val=gan_val, num_epoch=1)
