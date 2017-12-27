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


def build_transformer(opt, SRC_FIELD, TGT_FIELD):
    return Transformer(
        len(SRC_FIELD.vocab),
        len(TGT_FIELD.vocab),
        opt.max_token_seq_len,
        proj_share_weight=opt.proj_share_weight,
        embs_share_weight=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        encoder_emb_weight=SRC_FIELD.vocab.vectors,
        decoder_emb_weight=TGT_FIELD.vocab.vectors, )


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
    bidirectional = True
    encoder = EncoderRNN(len(EN.vocab), opt.max_len, hidden_size,
                         input_dropout_p=0, dropout_p=0, n_layers=1,
                         bidirectional=bidirectional, variable_lengths=True, rnn_cell='gru')
    decoder = DecoderRNN(len(EN.vocab), opt.max_len, hidden_size * 2 if bidirectional else 1, n_layers=1,
                         dropout_p=0.2, use_attention=True, bidirectional=bidirectional, rnn_cell='gru',
                         eos_id=Constants.EOS, sos_id=Constants.BOS)
    G = Seq2seq(encoder, decoder)
    for param in G.parameters():
        param.data.uniform_(-0.08, 0.08)

try:
    D = load_model(opt.load_D_from)
except AttributeError:
    D = BinaryClassifierCNN(len(EN.vocab),
                            embed_dim=opt.embed_dim,
                            num_kernel=opt.num_kernel,
                            kernel_sizes=opt.kernel_sizes,
                            dropout_p=opt.dropout_p)

# optim_G = ScheduledOptim(optim.Adam(
#     G.get_trainable_parameters(),
#     betas=(0.9, 0.98), eps=1e-09),
#     opt.d_model, opt.n_warmup_steps)
optim_G = optim.Adam(G.parameters(), lr=1e-4,
                     betas=(0.9, 0.98), eps=1e-09)
optim_D = torch.optim.Adam(D.parameters(), lr=1e-4)

crit_G = NLLLoss(size_average=False)
crit_D = nn.BCELoss()

if opt.cuda:
    G.cuda()
    D.cuda()
    crit_G.cuda()
    crit_D.cuda()

# ---------- train ----------

trainer_D = trainers.DiscriminatorTrainer()

# pre-train D
if not hasattr(opt, 'load_D_from'):
    pool = helper.DiscriminatorDataPool(opt.max_len, D.min_len, Constants.PAD)

    for epoch in range(1):
        logging.info('[Pretrain D Epoch %d]' % epoch)

        D.train()

        # 將資料塞進pool中
        train_iter = data.BucketIterator(
            dataset=train, batch_size=16, device=opt.device,
            sort_within_batch=True, sort_key=lambda x: len(x.src), repeat=False)

        for batch in train_iter:
            src_seq = batch.src[0]
            src_length = batch.src[1]
            # tgt_seq = src_seq.clone()

            decoder_outputs, decoder_hidden, other = G(
                src_seq, src_length.tolist(), target_variable=None)
            fake = torch.cat(other[DecoderRNN.KEY_SEQUENCE], dim=1)

            pool.append_fake(fake)
            pool.append_real(batch.tgt[0])

            if len(pool.fakes) > 500:
                break

        # pool.fill(train_iter)
        trainer_D.train(D, train_iter=pool.batch_gen(),
                        crit=crit_D, optimizer=optim_D)
        pool.reset()
    Checkpoint(model=D, optimizer=optim_D, epoch=0, step=0,
               input_vocab=EN.vocab, output_vocab=EN.vocab).save(opt._load_D_from)


def eval_D():
    pool = helper.DiscriminatorDataPool(opt.max_len, D.min_len, Constants.PAD)
    val_iter = data.BucketIterator(
        dataset=val, batch_size=opt.batch_size, device=opt.device,
        sort_key=lambda x: len(x.src), repeat=False)
    pool.fill(val_iter)
    trainer_D.evaluate(D, val_iter=pool.batch_gen(), crit=crit_D)

    # eval_D()


# pre-train G
if not hasattr(opt, 'load_G_from'):
    print('Pre-train G')
    trainer_G = SupervisedTrainer()
    trainer_G.optimizer = optim_G

    G.train()

    for epoch in range(5):
        train_iter = data.BucketIterator(
            dataset=train, batch_size=64, device=opt.device,
            sort_within_batch=True, sort_key=lambda x: len(x.src), repeat=False)
        for step, batch in enumerate(train_iter):
            src_seq = batch.src[0]
            src_length = batch.src[1]
            tgt_seq = src_seq.clone()

            # print(src_seq)

            loss = trainer_G._train_batch(
                src_seq, src_length.tolist(), tgt_seq, G, teacher_forcing_ratio=0)
            if step % 100 == 0:
                print('[step %d] loss_G %.4f' % (epoch * len(train_iter) + step, loss))
    Checkpoint(model=G, optimizer=optim_G, epoch=0, step=0,
               input_vocab=EN.vocab, output_vocab=EN.vocab).save(opt._load_G_from)

# Train SeqGAN
ALPHA = 0

for epoch in range(100):
    logging.info('[Epoch %d]' % epoch)
    train_iter = data.BucketIterator(
        dataset=train, batch_size=1, device=opt.device,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src), repeat=False)

    for step, batch in enumerate(train_iter):
        src_seq = batch.src[0]
        src_length = batch.src[1]
        tgt_seq = src_seq.clone()
        # gold = tgt_seq[:, 1:]

        # reconstruction loss
        # loss_G.reset()
        # decoder_outputs, decoder_hidden, other = G(src_seq, src_length.tolist(), target_variable=None)
        # fake = torch.cat(other[DecoderRNN.KEY_SEQUENCE], dim=1)

        # (1) train D
        # optim_G.zero_grad()
        # optim_D.zero_grad()
        #
        # D_real = D(src_seq)
        # D_fake = D(fake)
        # loss_D = -torch.mean(log(D_real) + log(1 - D_fake))
        #
        # loss_D.backward()
        # optim_D.step()

        # print(torch.cat(other[DecoderRNN.KEY_SEQUENCE], dim=1))
        # print(tgt_seq.data)
        # print(len(decoder_outputs) + 1)

        # if tgt_seq.size(1) < len(decoder_outputs) + 1:
        #     tgt_seq = helper.pad_seq(tgt_seq.data, max_len=len(decoder_outputs) + 1, pad_value=Constants.PAD)
        #     tgt_seq = autograd.Variable(tgt_seq)
        # for i, step_output in enumerate(decoder_outputs):
        #     batch_size = tgt_seq.size(0)
        #     loss_G.eval_batch(step_output.contiguous().view(batch_size, -1), tgt_seq[:, i + 1])

        # (2) train G
        D.eval()


        def rollout(G, D, other, n_rollout=8):
            rewards = []
            for i in range(0, opt.max_len):
                D_fake = None
                for j in range(n_rollout):
                    cur_seq = other[DecoderRNN.KEY_SEQUENCE][:i + 1]
                    _, _, ret_dict = G.rollout(
                        inputs=torch.cat(cur_seq, dim=1),
                        decoder_hidden=decoder_hidden[i],
                        encoder_outputs=encoder_outputs)
                    # print(len(ret_dict[DecoderRNN.KEY_SEQUENCE]))
                    fake = torch.cat(cur_seq + ret_dict[DecoderRNN.KEY_SEQUENCE], dim=1)
                    if D_fake is None:
                        D_fake = D(fake).unsqueeze(1)
                    else:
                        D_fake += D(fake).unsqueeze(1)
                        # print(D_fake)
                rewards.append(D_fake / float(n_rollout))
            return torch.cat(rewards, dim=1)


        encoder_outputs, decoder_outputs, decoder_hidden, other = G.sample(
            src_seq, src_length.tolist(), target_variable=None, n_sample=4)
        probs = torch.cat(other[DecoderRNN.KEY_PROB], dim=1).view(-1)  # (batch*len)
        rewards = rollout(G, D, other).view(-1)  # (batch*len)
        rewards = autograd.Variable(rewards.data, requires_grad=False)  # 避免grad error

        optim_G.zero_grad()
        loss_G = -torch.sum(torch.log(probs) * rewards)
        loss_G.backward()
        optim_G.step()

        if step % 100 == 0:
            ALPHA += 1E-5
            pred = torch.cat([x for x in other['sequence']], dim=1)
            # print('[step %d] loss_rest %.4f' % (epoch * len(train_iter) + step, loss_G.get_loss()))
            print('[step %d] loss_D %.4f, D_fake %.4f' % (
                epoch * len(train_iter) + step, loss_G.data[0], rewards.mean().data[0]))
            print('%s -> %s' % (EN.reverse(tgt_seq.data)[0], EN.reverse(pred.data)[0]))
