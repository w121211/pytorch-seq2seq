import logging
import os
import sys
import argparse
import copy
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
from torchtext import data
from torchtext import datasets

pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if pardir not in sys.path:
    sys.path.insert(0, pardir)

from seq2seq.util.checkpoint import Checkpoint
from seq2seq.trainer import SupervisedTrainer
from seq2seq.loss import NLLLoss

from ape import Constants, options, helper
from ape.dataset.lang8 import Lang8
from ape.dataset.field import SentencePieceField
from ape.model.discriminator import BinaryClassifierCNN
from ape.model.transformer.Models import Transformer
from ape.model.seq2seq import EncoderRNN, DecoderRNN, CycleSeq2seq
from ape import trainers

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'


def load_model(exp_path):
    cp = Checkpoint.load(Checkpoint.get_latest_checkpoint(exp_path))
    model = cp.model
    return model


parser = argparse.ArgumentParser()
parser = options.train_options(parser)
opt = parser.parse_args()

opt.cuda = torch.cuda.is_available()
opt.device = None if opt.cuda else -1

# 快速變更設定
opt.exp_dir = './experiment/transformer-reinforce/use_billion'
opt.load_vocab_from = './experiment/transformer/lang8-cor2err/vocab.pt'
opt.build_vocab_from = './data/billion/billion.30m.model.vocab'

opt._load_D_from = os.path.join(opt.exp_dir, 'pretrain_D')
opt._load_G_from = os.path.join(opt.exp_dir, 'pretrain_G')
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
                        include_lengths=True,
                        fix_length=opt.max_len + 1)
train = datasets.TranslationDataset(
    path='./data/dualgan/train',
    exts=('.billion.sp', '.use.sp'), fields=[('src', EN), ('tgt', EN)],
    filter_pred=len_filter)
val = datasets.TranslationDataset(
    path='./data/dualgan/val',
    exts=('.billion.sp', '.use.sp'), fields=[('src', EN), ('tgt', EN)],
    filter_pred=len_filter)
train_lang8, val_lang8 = Lang8.splits(
    exts=('.err.sp', '.cor.sp'), fields=[('src', EN), ('tgt', EN)],
    train='test', validation='test', test=None, filter_pred=len_filter)

# 讀取 vocabulary（確保一致）
try:
    logging.info('Load voab from %s' % opt.load_vocab_from)
    EN.load_vocab(opt.load_vocab_from)
except FileNotFoundError:
    EN.build_vocab_from(opt.build_vocab_from)
    EN.save_vocab(opt.load_vocab_from)

logging.info('Vocab len: %d' % len(EN.vocab))

# 檢查Constants是否有誤
assert EN.vocab.stoi[Constants.BOS_WORD] == Constants.BOS
assert EN.vocab.stoi[Constants.EOS_WORD] == Constants.EOS
assert EN.vocab.stoi[Constants.PAD_WORD] == Constants.PAD
assert EN.vocab.stoi[Constants.UNK_WORD] == Constants.UNK

# ---------- init model ----------

try:
    G = load_model(opt.load_G_from)
except AttributeError:
    hidden_size = 512
    bidirectional = False
    encoder_a = EncoderRNN(len(EN.vocab), opt.max_len, hidden_size,
                           input_dropout_p=0, dropout_p=0, n_layers=1,
                           bidirectional=bidirectional, variable_lengths=False, rnn_cell='gru')
    decoder_a = DecoderRNN(len(EN.vocab), opt.max_len,
                           hidden_size * 2 if bidirectional else hidden_size, n_layers=1,
                           dropout_p=0.2, use_attention=True, bidirectional=bidirectional, rnn_cell='gru',
                           eos_id=Constants.EOS, sos_id=Constants.BOS)
    encoder_b = copy.deepcopy(encoder_a)
    decoder_b = copy.deepcopy(decoder_a)
    G = CycleSeq2seq(encoder_a, decoder_a, encoder_b, decoder_b)
    for param in G.parameters():
        param.data.uniform_(-0.08, 0.08)

optim_G = optim.Adam(G.parameters(), lr=1e-4)
crit_G = NLLLoss(size_average=False)

if opt.cuda:
    G.cuda()
    crit_G.cuda()

# ---------- train ----------

for epoch in range(100):
    G.train()

    logging.info('[Epoch %d]' % epoch)
    train_iter = data.BucketIterator(
        dataset=train, batch_size=16, device=opt.device,
        sort_within_batch=True, sort_key=lambda x: len(x.src), repeat=False)

    for step, batch in enumerate(train_iter):
        src_seq = batch.src[0]
        src_length = batch.src[1]
        tgt_seq = src_seq.clone()  # a -> b' -> a

        decoder_outputs, decoder_hiddens, other = G.forward(
            src_seq, src_length.tolist(), target_variable=None)
        crit_G.reset()
        for i, step_output in enumerate(decoder_outputs):
            batch_size = tgt_seq.size(0)
            crit_G.eval_batch(step_output.contiguous().view(batch_size, -1), tgt_seq[:, i + 1])

        optim_G.zero_grad()
        crit_G.backward()
        optim_G.step()

        if step % 100 == 0:
            pred = torch.cat([x for x in other['sequence']], dim=1)
            print('[step %d] loss %.4f' % (
                epoch * len(train_iter) + step, crit_G.get_loss()))
            print('%s -> %s' % (EN.reverse(tgt_seq.data)[0], EN.reverse(pred.data)[0]))
