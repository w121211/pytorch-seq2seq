import logging
import os
import sys
import argparse
import copy
import random
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
from torchtext import data
from torchtext import datasets
from torchtext.vocab import FastText

pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if pardir not in sys.path:
    sys.path.insert(0, pardir)

from seq2seq.util.checkpoint import Checkpoint
from seq2seq.models import EncoderRNN
from seq2seq.trainer import SupervisedTrainer
from seq2seq.loss import NLLLoss

from ape import Constants, options, helper
from ape.dataset.lang8 import Lang8
from ape.dataset.field import SentencePieceField
from ape.model.discriminator import BinaryClassifierCNN
from ape.model.transformer.Models import Transformer
from ape.model.seq2seq import Seq2seq, DecoderRNN
from ape import trainers

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'


def load_model(exp_path):
    cp = Checkpoint.load(Checkpoint.get_latest_checkpoint(exp_path))
    model = cp.model
    return model


def to_noise(seq, length, p_o=0., p_x=0.5):
    '''生成noise seq。 https://arxiv.org/pdf/1602.03483.pdf
    Args:
        seq: input sequence (batch, max_len)
        length: length of sequence (batch, 1)
        p_o: seq中每一個word drop掉的機率（獨立）
        p_x: seq中每個字組(w_i, w_i+1)互換的機率
    '''
    lens = []
    rows = []

    for row, length in zip(seq.data.tolist(), length.tolist()):
        row = row[1:length - 1]  # 移除<BOS>, <EOS>, paddings

        # dropout
        drop_idx = [i for i in range(len(row)) if random.random() <= p_o]
        for i in sorted(drop_idx, reverse=True):
            del row[i]

        # swap
        len_iter = iter(range(len(row) - 1))
        for i in len_iter:
            if random.random() <= p_x:
                row[i], row[i + 1] = row[i + 1], row[i]
                next(len_iter, None)
        row.insert(0, Constants.BOS)
        row.append(Constants.EOS)
        rows.append(row)
        lens.append(len(row))

    # padding
    max_len = max(lens)
    for row in rows:
        while len(row) < max_len:
            row.append(Constants.PAD)

    noise = autograd.Variable(torch.LongTensor([rows])).type_as(seq)

    return noise


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser = options.train_options(parser)
opt = parser.parse_args()

opt.cuda = torch.cuda.is_available()
opt.device = None if opt.cuda else -1

# 快速變更設定
opt.exp_dir = './experiment/seq2seq/lang8'
# opt.load_vocab_from = './experiment/transformer/lang8-cor2err/vocab.pt'
# opt.build_vocab_from = './data/billion/billion.30m.model.vocab'

opt._load_D_from = os.path.join(opt.exp_dir, 'pretrain_D')
opt._load_G_a_from = os.path.join(opt.exp_dir, 'err2cor')
opt._load_G_b_from = os.path.join(opt.exp_dir, 'cor2err')
# opt.load_D_from = opt._load_D_from
# opt.load_G_from = opt._load_G_from

# dataset params
opt.max_len = 20

# G params
# opt.load_G_a_from = './experiment/transformer/lang8-err2cor/'
# opt.load_G_b_from = './experiment/transformer/lang8-cor2err/'
opt.d_word_vec = 300
opt.d_model = 300
opt.d_inner_hid = 600
opt.n_head = 6
opt.n_layers = 3
opt.embs_share_weight = False
opt.beam_size = 1
opt.max_token_seq_len = opt.max_len + 2  # 包含<BOS>, <EOS>
opt.n_warmup_steps = 4000

# D params
opt.embed_dim = opt.d_model
opt.num_kernel = 100
opt.kernel_sizes = [3, 4, 5, 6, 7]
opt.dropout_p = 0.25

# train params
opt.batch_size = 1
opt.n_epoch = 10

if not os.path.exists(opt.exp_dir):
    os.makedirs(opt.exp_dir)
logging.basicConfig(filename=opt.exp_dir + '/.log',
                    format=LOG_FORMAT, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

logging.info('Use CUDA? ' + str(opt.cuda))
logging.info(opt)


# ---------- prepare dataset ----------

def len_filter(example):
    return len(example.src) <= opt.max_len and len(example.tgt) <= opt.max_len


EN = SentencePieceField(init_token=Constants.BOS_WORD,
                        eos_token=Constants.EOS_WORD,
                        batch_first=True,
                        include_lengths=True, )  # fix_length=opt.max_len + 1)
train = datasets.TranslationDataset(
    path='./data/dualgan/train',
    exts=('.billion.sp', '.use.sp'), fields=[('src', EN), ('tgt', EN)],
    filter_pred=len_filter)
# val = datasets.TranslationDataset(
#     path='./data/dualgan/val',
#     exts=('.billion.sp', '.use.sp'), fields=[('src', EN), ('tgt', EN)],
#     filter_pred=len_filter)
train_lang8_err2cor, _ = Lang8.splits(
    exts=('.err', '.cor'), fields=[('src', EN), ('tgt', EN)],
    train='test', validation='test', test=None, filter_pred=len_filter)
train_lang8_cor2err, _ = Lang8.splits(
    exts=('.err', '.cor'), fields=[('src', EN), ('tgt', EN)],
    train='test', validation='test', test=None, filter_pred=len_filter)

# 讀取 vocabulary（確保一致）
try:
    logging.info('Load voab from %s' % opt.load_vocab_from)
    EN.load_vocab(opt.load_vocab_from)
except (FileNotFoundError, AttributeError):
    # EN.build_vocab_from(opt.build_vocab_from)
    # EN.save_vocab(opt.load_vocab_from)
    EN.build_vocab(train, train_lang8_err2cor, train_lang8_cor2err,
                   vectors=[FastText(language="en")], max_size=20000, min_freq=2)

logging.info('Vocab len: %d' % len(EN.vocab))

# 檢查Constants是否有誤
assert EN.vocab.stoi[Constants.BOS_WORD] == Constants.BOS
assert EN.vocab.stoi[Constants.EOS_WORD] == Constants.EOS
assert EN.vocab.stoi[Constants.PAD_WORD] == Constants.PAD
assert EN.vocab.stoi[Constants.UNK_WORD] == Constants.UNK

# ---------- init model ----------

try:
    G_a = load_model(opt.load_G_a_from)
except AttributeError:
    hidden_size = 300
    bidirectional = False
    encoder = EncoderRNN(len(EN.vocab), opt.max_len, hidden_size,
                         input_dropout_p=0, dropout_p=0, n_layers=1,
                         bidirectional=bidirectional, variable_lengths=False, rnn_cell='gru')
    encoder.embedding.weight = nn.Parameter(EN.vocab.vectors)
    encoder.embedding.weight.requires_grad = False  # 不更新embedding weights

    decoder_a = DecoderRNN(len(EN.vocab), opt.max_len, hidden_size * 2 if bidirectional else hidden_size, n_layers=1,
                           dropout_p=0.2, use_attention=True, bidirectional=bidirectional, rnn_cell='gru',
                           eos_id=Constants.EOS, sos_id=Constants.BOS)
    decoder_a.embedding.weight = nn.Parameter(EN.vocab.vectors)
    decoder_a.embedding.weight.requires_grad = False  # 不更新embedding weights

    G_a = Seq2seq(encoder, decoder_a)
    # for param in G_a.parameters():
    #     param.data.uniform_(-0.08, 0.08)
try:
    G_b = load_model(opt.load_G_b_from)
except AttributeError:
    G_b = Seq2seq(G_a.encoder, copy.deepcopy(G_a.decoder))  # 共用同一個encoder
    # for param in G_b.parameters():
    #     param.data.uniform_(-0.08, 0.08)

# optim_G = ScheduledOptim(optim.Adam(
#     G.get_trainable_parameters(),
#     betas=(0.9, 0.98), eps=1e-09),
#     opt.d_model, opt.n_warmup_steps)

optim_G_a = optim.Adam(filter(lambda p: p.requires_grad, G_a.parameters()), lr=1e-4,
                       betas=(0.9, 0.98), eps=1e-09)
optim_G_b = optim.Adam(filter(lambda p: p.requires_grad, G_b.parameters()), lr=1e-4,
                       betas=(0.9, 0.98), eps=1e-09)
crit_G = NLLLoss(size_average=False)

if opt.cuda:
    G_a.cuda()
    G_b.cuda()
    crit_G.cuda()

trainer_G_a = SupervisedTrainer()
trainer_G_b = SupervisedTrainer()
trainer_G_a.optimizer = optim_G_a
trainer_G_b.optimizer = optim_G_b


# ---------- train ----------

def pretrain(model, trainer, dataset):
    model.train()
    for epoch in range(20):
        print('\n[Epoch %d]' % epoch)
        train_iter = data.BucketIterator(
            dataset=dataset, batch_size=64, device=opt.device,
            sort_within_batch=True, sort_key=lambda x: len(x.src), repeat=False)
        for step, batch in enumerate(train_iter):
            src_seq = batch.src[0]
            src_length = batch.src[1]
            tgt_seq = batch.tgt[0]
            loss = trainer._train_batch(
                src_seq, src_length.tolist(), tgt_seq, model, teacher_forcing_ratio=0)
            if step % 100 == 0:
                print('[step %d] loss_G %.4f' % (epoch * len(train_iter) + step, loss))


if not hasattr(opt, 'load_G_a_from'):
    print('Pre-train G_a')
    pretrain(G_a, trainer_G_a, train_lang8_err2cor)
    Checkpoint(model=G_a, optimizer=trainer_G_a.optimizer, epoch=0, step=0,
               input_vocab=EN.vocab, output_vocab=EN.vocab).save(opt._load_G_a_from)

if not hasattr(opt, 'load_G_b_from'):
    print('Pre-train G_b')
    pretrain(G_b, trainer_G_a, train_lang8_cor2err)
    Checkpoint(model=G_b, optimizer=trainer_G_b.optimizer, epoch=0, step=0,
               input_vocab=EN.vocab, output_vocab=EN.vocab).save(opt._load_G_b_from)

# train
print('Dual-train')
if not hasattr(opt, 'load_G_from'):
    for epoch in range(20):
        G_a.train()
        G_b.train()

        print('\n[Epoch %d]' % epoch)
        train_iter = data.BucketIterator(
            dataset=train, batch_size=32, device=opt.device,
            sort_within_batch=True, sort_key=lambda x: len(x.src), repeat=False)
        for step, batch in enumerate(train_iter):
            real_a = batch.src[0]
            real_a_len = batch.src[1]
            real_b = batch.tgt[0]
            real_b_len = batch.tgt[1]

            if step % 2 == 0:
                # (1) denoising
                noise_b = to_noise(real_b, real_b_len, p_o=0., p_x=0.5)
                denoise_loss_G_a = trainer_G_a._train_batch(
                    noise_b, None, real_b, G_a, teacher_forcing_ratio=0)

                noise_a = to_noise(real_a, real_a_len, p_o=0., p_x=0.5)
                denoise_loss_G_b = trainer_G_b._train_batch(
                    noise_a, None, real_a, G_b, teacher_forcing_ratio=0)
            else:
                # (2) back-translation
                def to_seq(decode_list):
                    lengths = []
                    for seq in decode_list:
                        seq = seq[:seq.index(Constants.EOS)]
                        seq.insert(0, Constants.BOS)
                        lengths.append(len(seq))

                    max_len = max(lengths)
                    for seq in decode_list:
                        while len(seq) < max_len:
                            seq.append(Constants.PAD)


                G_a.eval()
                G_b.train()
                _, _, other = G_a.forward(real_a, real_a_len.tolist())  # a -> fake_b
                fake_b = to_seq(other[DecoderRNN.KEY_SEQUENCE])
                loss_rest_G_b = trainer_G_b._train_batch(fake_b, None, real_a, G_b, teacher_forcing_ratio=0)

                G_b.eval()
                G_a.train()
                _, _, other = G_b.forward(real_b, real_b_len.tolist())  # a -> fake_b
                fake_a = to_seq(other[DecoderRNN.KEY_SEQUENCE])
                loss_rest_G_a = trainer_G_a._train_batch(fake_a, None, real_b, G_a, teacher_forcing_ratio=0)
            if step % 100 == 0:
                print('[step %d] denoise_loss_G_a %.4f, denoise_loss_G_b % .4f' % (
                    epoch * len(train_iter) + step, denoise_loss_G_a, denoise_loss_G_b))
            if step % 101 == 0:
                print('[step %d] loss_rest_G_a %.4f, loss_rest_G_b % .4f' % (
                    epoch * len(train_iter) + step, loss_rest_G_a, loss_rest_G_b))

        Checkpoint(model=G_a, optimizer=optim_G_a, epoch=0, step=0,
                   input_vocab=EN.vocab, output_vocab=EN.vocab).save(opt._load_G_a_from)
        Checkpoint(model=G_b, optimizer=optim_G_b, epoch=0, step=0,
                   input_vocab=EN.vocab, output_vocab=EN.vocab).save(opt._load_G_b_from)
