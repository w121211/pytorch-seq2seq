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
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.models import EncoderRNN
from seq2seq.loss import NLLLoss

from ape import Constants, options, helper
from ape.dataset.lang8 import Lang8
from ape.dataset.field import SentencePieceField
from ape.model.discriminator import BinaryClassifierCNN
from ape.model.transformer.Models import Transformer
from ape.model.transformer.Optim import ScheduledOptim
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


def build_D(opt, TEXT_FIELD):
    return BinaryClassifierCNN(len(TEXT_FIELD.vocab),
                               embed_dim=opt.embed_dim,
                               num_kernel=opt.num_kernel,
                               kernel_sizes=opt.kernel_sizes,
                               dropout_p=opt.dropout_p)


def load_model(exp_path):
    cp = Checkpoint.load(Checkpoint.get_latest_checkpoint(exp_path))
    model = cp.model
    return model


def main():
    parser = argparse.ArgumentParser()
    opt = options.train_options(parser)
    opt = parser.parse_args()

    opt.cuda = torch.cuda.is_available()
    opt.device = None if opt.cuda else -1

    # 快速變更設定
    opt.exp_dir = './experiment/transformer-reinforce/use_billion'
    opt.load_vocab_from = './experiment/transformer/lang8-cor2err/vocab.pt'
    opt.build_vocab_from = './data/billion/billion.30m.model.vocab'

    opt.load_D_from = opt.exp_dir
    # opt.load_D_from = None

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
                            include_lengths=True)

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

    # G = build_G(opt, EN, EN)
    hidden_size = 512
    bidirectional = True
    encoder = EncoderRNN(len(EN.vocab), opt.max_len, hidden_size, n_layers=1,
                         bidirectional=bidirectional)
    decoder = DecoderRNN(len(EN.vocab), opt.max_len, hidden_size * 2 if bidirectional else 1, n_layers=1,
                         dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
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
    loss_G = NLLLoss(size_average=False)
    if torch.cuda.is_available():
        loss_G.cuda()

    # # 預先訓練D
    if opt.load_D_from:
        D = load_model(opt.load_D_from)
    else:
        D = build_D(opt, EN)
    optim_D = torch.optim.Adam(D.parameters(), lr=1e-4)

    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit_G = get_criterion(len(EN.vocab))
    crit_D = nn.BCELoss()

    if opt.cuda:
        G.cuda()
        D.cuda()
        crit_G.cuda()
        crit_D.cuda()

    # ---------- train ----------

    trainer_D = trainers.DiscriminatorTrainer()

    if not opt.load_D_from:
        for epoch in range(1):
            logging.info('[Pretrain D Epoch %d]' % epoch)

            pool = helper.DiscriminatorDataPool(opt.max_len, D.min_len, Constants.PAD)

            # 將資料塞進pool中
            train_iter = data.BucketIterator(
                dataset=train, batch_size=opt.batch_size, device=opt.device,
                sort_key=lambda x: len(x.src), repeat=False)
            pool.fill(train_iter)

            # train D
            trainer_D.train(D, train_iter=pool.batch_gen(),
                            crit=crit_D, optimizer=optim_D)
            pool.reset()

        Checkpoint(model=D, optimizer=optim_D, epoch=0, step=0,
                   input_vocab=EN.vocab, output_vocab=EN.vocab).save(opt.exp_dir)

    def eval_D():
        pool = helper.DiscriminatorDataPool(opt.max_len, D.min_len, Constants.PAD)
        val_iter = data.BucketIterator(
            dataset=val, batch_size=opt.batch_size, device=opt.device,
            sort_key=lambda x: len(x.src), repeat=False)
        pool.fill(val_iter)
        trainer_D.evaluate(D, val_iter=pool.batch_gen(), crit=crit_D)

        # eval_D()

    # Train G
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
            tgt_seq = src_seq[0].clone()
            # gold = tgt_seq[:, 1:]


            optim_G.zero_grad()
            loss_G.reset()

            decoder_outputs, decoder_hidden, other = G.rollout(src_seq, None, None, n_rollout=1)
            for i, step_output in enumerate(decoder_outputs):
                batch_size = tgt_seq.size(0)
                # print(step_output)

                # loss_G.eval_batch(step_output.contiguous().view(batch_size, -1), tgt_seq[:, i + 1])

            softmax_output = torch.exp(torch.cat([x for x in decoder_outputs], dim=0)).unsqueeze(0)
            softmax_output = helper.stack(softmax_output, 8)

            print(softmax_output)
            rollout = softmax_output.multinomial(1)
            print(rollout)

            tgt_seq = helper.pad_seq(tgt_seq.data, max_len=len(decoder_outputs) + 1, pad_value=Constants.PAD)
            tgt_seq = autograd.Variable(tgt_seq)
            for i, step_output in enumerate(decoder_outputs):
                batch_size = tgt_seq.size(0)
                loss_G.eval_batch(step_output.contiguous().view(batch_size, -1), tgt_seq[:, i + 1])
            G.zero_grad()
            loss_G.backward()
            optim_G.step()

            if step % 100 == 0:
                pred = torch.cat([x for x in other['sequence']], dim=1)
                print('[step %d] loss_rest %.4f' % (epoch * len(train_iter) + step, loss_G.get_loss()))
                print('%s -> %s' % (EN.reverse(tgt_seq.data)[0], EN.reverse(pred.data)[0]))

    # Reinforce Train G
    for p in D.parameters():
        p.requires_grad = False

        # for epoch in range(10):
        #     logging.info('\n\n[Epoch %d]' % epoch)
        #
        #     train_iter = data.BucketIterator(
        #         dataset=train, batch_size=1, device=opt.device,
        #         sort_key=lambda x: len(x.src), repeat=False)
        #
        #     for step, batch in enumerate(train_iter):
        #         src_seq = batch.src
        #         # tgt_seq = batch.tgt
        #         tgt_seq = src_seq.clone()
        #         gold = tgt_seq[:, 1:]
        #
        #         optim_G.zero_grad()
        #
        #         # D reward，產生的seq越像B，D給予的機率越高
        #         # stacked_src_seq = helper.stack(src_seq, 8)
        #         # hyp_seq, softmax_outputs = G.translate(stacked_src_seq)
        #         # # print(hyp_seq)
        #         #
        #         # log_probs = D(hyp_seq).log().unsqueeze(1)
        #         # reward_D = torch.mul(softmax_outputs, log_probs).mean()  # reward from D
        #
        #         # 還原度
        #         logit = G(src_seq, tgt_seq)
        #         pred = logit.view(-1, logit.size(2))  # (batch*len, n_vocab)
        #         loss_rest = crit_G(pred, gold.contiguous().view(-1))
        #
        #         # loss = 0.9 * loss_rest + 0.1 * -reward_D
        #         loss = loss_rest
        #
        #         loss.backward()
        #         optim_G.step()
        #         if step % 20 == 0:
        #             optim_G.update_learning_rate()
        #
        #         if step % 100 == 0:
        #             pred = pred.max(1)[1].view(src_seq.size(0), -1)
        #             # print('[step %d] loss_pg %.4f, loss_rest %.4f' % (step, reward_D.data[0], loss_rest.data[0]))
        #             print('[step %d] loss_rest %.4f' % (step, loss_rest.data[0]))
        #             print('%s -> %s' % (EN.reverse(gold.data)[0], EN.reverse(pred.data)[0]))


def train_transformer():
    for epoch in range(10):
        logging.info('[Epoch %d]' % epoch)

        train_iter = data.BucketIterator(
            dataset=train, batch_size=16, device=opt.device,
            sort_key=lambda x: len(x.src), repeat=False)

        for step, batch in enumerate(train_iter):
            src_seq = batch.src
            tgt_seq = src_seq.clone()
            gold = tgt_seq[:, 1:]

            optim_G.zero_grad()
            logit = G(src_seq, tgt_seq)
            pred = logit.view(-1, logit.size(2))  # (batch*len, n_vocab)
            loss_rest = crit_G(pred, gold.contiguous().view(-1))
            loss = loss_rest

            # print(loss_pg)

            loss.backward()
            optim_G.step()
            # optim_G.update_learning_rate()

            if step % 100 == 0:
                pred = pred.max(1)[1].view(src_seq.size(0), -1)
                print('[step %d] loss_rest %.4f' % (step, loss_rest.data[0]))
                print('%s -> %s' % (EN.reverse(gold.data)[0], EN.reverse(pred.data)[0]))


def PGLoss(G, D, src_seq):
    ''' Policy gradient training on G '''
    src_pos = G.get_position(src_seq.data)
    enc_output, *_ = G.encoder(src_seq, src_pos)
    dec_seq = autograd.Variable(torch.LongTensor(1, 1).fill_(Constants.BOS))
    if torch.cuda.is_available():
        dec_seq = dec_seq.cuda()

    rewards = None
    probs = None

    # decode
    for i in range(G.max_len):
        rollout_tokens, prob = G.step_rollout(src_seq, enc_output, dec_seq,
                                              n_rollout=6)
        rollout_tokens = rollout_tokens.transpose(1, 0)  # (n_rollout, 1)

        partial_seq = helper.stack(dec_seq.data, 6)  # (n_rollout, cur_len)
        partial_seq = torch.cat([partial_seq, rollout_tokens], dim=1)  # (n_rollout, cur_len+1)
        if partial_seq.size(1) < D.min_len:
            partial_seq = helper.pad_seq(partial_seq, D.min_len, Constants.PAD)

        partial_seq = autograd.Variable(partial_seq)

        reward = D(partial_seq)

        top_i = reward.max(dim=0)[1].data
        next_token = rollout_tokens.squeeze(1)[top_i]  # 選reward最高的為下個token
        next_token = autograd.Variable(next_token.unsqueeze(1))  # 需轉為variable，torch.cat()才不會出錯

        dec_seq = torch.cat([dec_seq, next_token], dim=1)

        rewards = torch.cat([rewards, reward]) if rewards is not None else reward
        probs = torch.cat([probs, prob]) if probs is not None else reward
        # probs += list(prob.split(1))

        if next_token[0] == Constants.EOS:
            break

    # print(rewards)
    loss = -torch.mean(rewards * probs)

    return loss


def StepPGLoss(G, D, src_seq):
    ''' Policy gradient training on G '''
    src_pos = G.get_position(src_seq.data)
    enc_output, *_ = G.encoder(src_seq, src_pos)
    dec_seq = autograd.Variable(torch.LongTensor(1, 1).fill_(Constants.BOS))
    if torch.cuda.is_available():
        dec_seq = dec_seq.cuda()

    rewards = None
    probs = None

    # decode
    for i in range(G.max_len):
        rollout_tokens, prob = G.step_rollout(src_seq, enc_output, dec_seq,
                                              n_rollout=6)
        rollout_tokens = rollout_tokens.transpose(1, 0)  # (n_rollout, 1)

        partial_seq = helper.stack(dec_seq.data, 6)  # (n_rollout, cur_len)
        partial_seq = torch.cat([partial_seq, rollout_tokens], dim=1)  # (n_rollout, cur_len+1)
        if partial_seq.size(1) < D.min_len:
            partial_seq = helper.pad_seq(partial_seq, D.min_len, Constants.PAD)

        partial_seq = autograd.Variable(partial_seq)

        reward = D(partial_seq)

        top_i = reward.max(dim=0)[1].data
        next_token = rollout_tokens.squeeze(1)[top_i]  # 選reward最高的為下個token
        next_token = autograd.Variable(next_token.unsqueeze(1))  # 需轉為variable，torch.cat()才不會出錯

        dec_seq = torch.cat([dec_seq, next_token], dim=1)

        rewards = torch.cat([rewards, reward]) if rewards is not None else reward
        probs = torch.cat([probs, prob]) if probs is not None else reward
        # probs += list(prob.split(1))

        if next_token[0] == Constants.EOS:
            break

    # print(rewards)
    loss = -torch.mean(rewards * probs)

    return loss


if __name__ == '__main__':
    main()
