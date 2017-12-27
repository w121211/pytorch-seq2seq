import logging
import os
import sys
import argparse
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


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser = options.train_options(parser)
opt = parser.parse_args()

opt.cuda = torch.cuda.is_available()
opt.device = None if opt.cuda else -1

# 快速變更設定
opt.exp_dir = './experiment/seq2seq/lang8'
opt.load_vocab_from = './experiment/transformer/lang8-cor2err/vocab.pt'
opt.build_vocab_from = './data/billion/billion.30m.model.vocab'

opt._load_D_from = os.path.join(opt.exp_dir, 'pretrain_D')
opt._load_G_from = os.path.join(opt.exp_dir, 'cor2err')
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
# train = datasets.TranslationDataset(
#     path='./data/dualgan/train',
#     exts=('.billion.sp', '.use.sp'), fields=[('src', EN), ('tgt', EN)],
#     filter_pred=len_filter)
# val = datasets.TranslationDataset(
#     path='./data/dualgan/val',
#     exts=('.billion.sp', '.use.sp'), fields=[('src', EN), ('tgt', EN)],
#     filter_pred=len_filter)
train_lang8, val_lang8 = Lang8.splits(
    exts=('.cor.sp', '.err.sp'), fields=[('src', EN), ('tgt', EN)],
    train='train', validation='val', test=None, filter_pred=len_filter)

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
    encoder = EncoderRNN(len(EN.vocab), opt.max_len, hidden_size,
                         input_dropout_p=0, dropout_p=0, n_layers=1,
                         bidirectional=bidirectional, variable_lengths=False, rnn_cell='gru')
    decoder = DecoderRNN(len(EN.vocab), opt.max_len, hidden_size * 2 if bidirectional else hidden_size, n_layers=1,
                         dropout_p=0.2, use_attention=True, bidirectional=bidirectional, rnn_cell='gru',
                         eos_id=Constants.EOS, sos_id=Constants.BOS)
    G = Seq2seq(encoder, decoder)
    for param in G.parameters():
        param.data.uniform_(-0.08, 0.08)

# optim_G = ScheduledOptim(optim.Adam(
#     G.get_trainable_parameters(),
#     betas=(0.9, 0.98), eps=1e-09),
#     opt.d_model, opt.n_warmup_steps)
optim_G = optim.Adam(G.parameters(), lr=1e-4,
                     betas=(0.9, 0.98), eps=1e-09)

crit_G = NLLLoss(size_average=False)

if opt.cuda:
    G.cuda()
    crit_G.cuda()

# ---------- train ----------

if not hasattr(opt, 'load_G_from'):
    print('Pre-train G')
    trainer_G = SupervisedTrainer()
    trainer_G.optimizer = optim_G

    G.train()

    for epoch in range(20):
        print('\n[Epoch %d]' % epoch)
        train_iter = data.BucketIterator(
            dataset=train_lang8, batch_size=64, device=opt.device,
            sort_within_batch=True, sort_key=lambda x: len(x.src), repeat=False)
        for step, batch in enumerate(train_iter):
            src_seq = batch.src[0]
            src_length = batch.src[1]
            tgt_seq = batch.tgt[0]

            loss = trainer_G._train_batch(
                src_seq, src_length.tolist(), tgt_seq, G, teacher_forcing_ratio=0)
            if step % 100 == 0:
                print('[step %d] loss_G %.4f' % (epoch * len(train_iter) + step, loss))

    Checkpoint(model=G, optimizer=optim_G, epoch=0, step=0,
               input_vocab=EN.vocab, output_vocab=EN.vocab).save(opt._load_G_from)
