import argparse
import logging
import os
import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torchtext.data as data
from torchtext.vocab import GloVe, CharNGram, FastText
from tqdm import tqdm

pardir = os.path.realpath(os.path.join(os.path.abspath(__file__), '../../..'))
if pardir not in sys.path:
    sys.path.insert(0, pardir)

from ape import helper, Constants
from ape.dataset.lang8 import Lang8
from ape.model.discriminator import DiscriminatorCNN
from ape.model.transformer.Models import Transformer
from ape.trainer.supervised import SupervisedDiscriminatorTrainer


class DualGanPGTrainer(object):
    train_D_every = 128  # 每train_G 128次，train_D 1次
    print_every = 1000

    n_rollout = 8

    lambda_a = 1
    lambda_b = 1

    def __init__(self, opt, trainer_D):
        self.logger = logging.getLogger(__name__)

        self.opt = opt
        self.trainer_D = trainer_D

    def train_G_PG(self, G, D, optim_G, src_seq):
        ''' Policy gradient training on G '''
        # TODO: add beam?
        for p in D.parameters():
            p.requires_grad = False

        # intermediate D reward
        # Dual training有將還原度一併作為reward，這邊暫時不考慮
        optim_G.zero_grad()

        src_pos = G.get_position(src_seq.data)
        enc_output, *_ = G.encoder(src_seq, src_pos)
        dec_seq = autograd.Variable(torch.LongTensor(1, 1).fill_(Constants.BOS))
        if torch.cuda.is_available():
            dec_seq = dec_seq.cuda()

        rewards = None
        probs = None
        for i in range(G.max_len):
            rollout_tokens, prob = G.step_rollout(src_seq, enc_output, dec_seq,
                                                  n_rollout=self.n_rollout)
            rollout_tokens = rollout_tokens.transpose(1, 0)  # (n_rollout, 1)

            partial_seq = helper.stack_seq(dec_seq.data, self.n_rollout)  # (n_rollout, cur_len)
            partial_seq = torch.cat([partial_seq, rollout_tokens], dim=1)  # (n_rollout, cur_len+1)
            if partial_seq.size(1) < D.min_len:
                partial_seq = helper.pad_seq(partial_seq, D.min_len, Constants.PAD)

            reward = D(partial_seq)

            top_i = reward.max(dim=0)[1].data
            next_token = rollout_tokens.squeeze(1)[top_i]  # 選reward最高的為下個token
            dec_seq = torch.cat([dec_seq, next_token.unsqueeze(1)], dim=1)

            rewards = torch.cat([rewards, reward]) if rewards is not None else reward
            probs = torch.cat([probs, prob]) if probs is not None else reward
            # probs += list(prob.split(1))

            if next_token[0] == Constants.EOS:
                break

        # back propagation
        loss = -torch.mean(rewards * probs)
        loss.backward()
        nn.utils.clip_grad_norm(G.get_trainable_parameters(), 40)  # 避免grad爆炸
        optim_G.step()

        return dec_seq

    def train(self, epoch, train_iter, G_a, G_b, D_a, D_b,
              optim_G_a, optim_G_b, optim_D_a, optim_D_b, crit_G, crit_D):
        D_a.train()
        D_b.train()
        G_a.train()
        G_b.train()

        pool_a = helper.DiscriminatorDataPool(G_a.n_max_seq, Constants.PAD)
        pool_b = helper.DiscriminatorDataPool(G_b.n_max_seq, Constants.PAD)

        step = len(train_iter) * epoch
        for batch in tqdm(train_iter):
            real_a = batch.src
            real_b = batch.tgt
            batch_size = real_a.size(0)

            # (1) train G
            # (1.1) realness of G by policy gradient
            fake_b = self.train_G_PG(G_a, D_b, optim_G_a, src_seq=real_a)
            fake_a = self.train_G_PG(G_b, D_a, optim_G_b, src_seq=real_b)

            # (1.2) cycle loss
            optim_G_a.zero_grad()
            optim_G_b.zero_grad()

            fake_a = autograd.Variable(fake_a.data)
            fake_b = autograd.Variable(fake_b.data)
            if self.opt.cuda:
                fake_a.cuda()
                fake_b.cuda()

            logit = G_a(real_a, fake_b)  # 重跑一遍獲得gradients
            _fake_b = G_a.decode_seq(logit)
            rest_a_logit = G_b(_fake_b, real_a)  # 使用teacher-forcing
            # rest_a = G_b.translate(fake_b)  # 不使用teacher-forcing

            logit = G_b(real_b, fake_a)
            _fake_a = G_b.decode_seq(logit)
            rest_b_logit = G_a(_fake_a, real_b)

            # 若非採用teacher-forcing時，需考慮2種情況
            # 若pred_len > gold_len，pad gold
            # 若pred_len < gold_len，pad one-hot（因為pred是以機率方式呈現）
            # pred = pred.view(-1, pred.size(2))  # (batch*len, n_vocab)
            # loss = crit(pred, gold.contiguous().view(-1))

            # print(rest_a_logit)
            # print(real_a.contiguous().view(-1))

            def criterion(crit, pred, gold):
                gold = gold[:, 1:]  # 去掉<bos>
                pred = pred.view(-1, pred.size(2))  # (batch*len, n_vocab)
                return crit(pred, gold.contiguous().view(-1))

            loss_cycle_a = criterion(crit_G, rest_a_logit, real_a)
            loss_cycle_b = criterion(crit_G, rest_b_logit, real_b)

            loss_cycle = self.lambda_a * loss_cycle_a + self.lambda_b * loss_cycle_b
            loss_cycle.backward()

            optim_G_a.step()
            optim_G_b.step()

            pool_a.append_fake(fake_a, to_partial=True)
            pool_a.append_real(real_a, to_partial=True)
            pool_b.append_fake(fake_b, to_partial=True)
            pool_b.append_real(real_b, to_partial=True)

            batch = next(pool_a.batch_gen())
            # print(batch.seq)
            # print(batch.label)
            # print([(batch.seq, batch.label) for batch in pool_a.batch_gen()])

            # (2) train D
            if step % self.train_D_every == 0:
                self.logger.info('Step[%d] Train D', step)
                for p in D_a.parameters():
                    p.requires_grad = True
                for p in D_b.parameters():
                    p.requires_grad = True
                self.trainer_D.train(D_a,
                                     train_iter=pool_a.batch_gen(),
                                     crit=crit_D,
                                     optimizer=optim_D_a)
                self.trainer_D.train(D_b,
                                     train_iter=pool_b.batch_gen(),
                                     crit=crit_D,
                                     optimizer=optim_D_b)
                # self.logger.info('rewards_a: avg %.4f, var %.4f' % (rewards_a.mean().data[0], rewards_a.var().data[0]))
                # self.logger.info('rewards_b: avg %.4f, var %.4f' % (rewards_b.mean().data[0], rewards_b.var().data[0]))
                # self.logger.info('loss_pg_a: %.4f' % loss_pg_a.data[0])
                # self.logger.info('loss_pg_b: %.4f' % loss_pg_b.data[0])
                pool_a.reset()
                pool_b.reset()

            if step % self.print_every == 0:
                # self.logger.info('loss g: %.4f (%d/%d)' % (loss_G.data[0], step, len(train_iter) * batch_size))
                pass
            step += 1


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('-data', required=True)

    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-resume', action='store_true')

    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                            decoded sentences""")

    opt = parser.parse_args()
    opt.d_word_vec = opt.d_model
    opt.cuda = torch.cuda.is_available()

    # 快速變更設定
    opt.batch_size = 1
    opt.epoch = 10

    opt.d_word_vec = 300
    opt.d_model = 300
    opt.d_inner_hid = 600

    opt.n_head = 5
    opt.n_layers = 3

    opt.embs_share_weight = True

    opt.beam_size = 1

    opt.max_len = 10
    opt.min_len = 5
    opt.max_token_seq_len = opt.max_len + 2  # 包含<BOS>, <EOS>

    opt.device = None if torch.cuda.is_available() else -1

    # ---------- prepare dataset ----------

    def len_filter(example):
        return len(example.src) <= opt.max_len and len(example.tgt) <= opt.max_len

    EN = data.ReversibleField(init_token=Constants.BOS_WORD,
                              eos_token=Constants.EOS_WORD,
                              batch_first=True)
    train, val = Lang8.splits(
        exts=('.err.bpe', '.cor.bpe'), fields=[('src', EN), ('tgt', EN)],
        train='test.small', validation='test.small', test=None, filter_pred=len_filter)
    # EN.build_vocab(train, vectors=[GloVe(name='840B', dim='300'), CharNGram(), FastText()])
    EN.build_vocab(train)
    logging.info('vocab len: %d' % len(EN.vocab))

    # 檢查Constants是否有誤
    assert EN.vocab.stoi[EN.init_token] == Constants.BOS
    assert EN.vocab.stoi[EN.eos_token] == Constants.EOS
    assert EN.vocab.stoi[EN.pad_token] == Constants.PAD
    assert EN.vocab.stoi[EN.unk_token] == Constants.UNK

    # ---------- init model ----------

    logging.info(opt)

    transformer_a = Transformer(
        len(EN.vocab),
        len(EN.vocab),
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
        encoder_emb_weight=EN.vocab.vectors,
        decoder_emb_weight=EN.vocab.vectors, )
    transformer_b = Transformer(
        len(EN.vocab),
        len(EN.vocab),
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
        encoder_emb_weight=EN.vocab.vectors,
        decoder_emb_weight=EN.vocab.vectors, )
    discriminator_a = DiscriminatorCNN(
        len(EN.vocab),
        min_len=opt.min_len,
        embed_dim=opt.d_word_vec,
        num_kernel=100,
        kernel_sizes=[2, 3, 4, 5],
        dropout_p=0.2)
    discriminator_b = DiscriminatorCNN(
        len(EN.vocab),
        min_len=opt.min_len,
        embed_dim=opt.d_word_vec,
        num_kernel=100,
        kernel_sizes=[2, 3, 4, 5],
        dropout_p=0.2)

    # print(transformer)

    optim_G_a = optim.Adam(transformer_a.get_trainable_parameters(),
                           betas=(0.9, 0.98), eps=1e-09)
    optim_G_b = optim.Adam(transformer_a.get_trainable_parameters(),
                           betas=(0.9, 0.98), eps=1e-09)
    optim_D_a = optim.Adam(discriminator_a.parameters(), lr=1e-3)
    optim_D_b = optim.Adam(discriminator_a.parameters(), lr=1e-3)

    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit_G = get_criterion(len(EN.vocab))
    crit_D = nn.BCELoss()

    if opt.cuda:
        transformer_a.cuda()
        transformer_b.cuda()
        discriminator_a.cuda()
        discriminator_b.cuda()
        crit_G.cuda()

    # ---------- training ----------

    train_iter, val_iter = data.BucketIterator.splits(
        (train, val), batch_sizes=(opt.batch_size, 128), device=opt.device,
        sort_key=lambda x: len(x.src), repeat=False)

    batch = next(iter(train_iter))
    src_seq = batch.src
    tgt_seq = batch.tgt

    trinaer_D = SupervisedDiscriminatorTrainer()

    trainer = DualGanPGTrainer(opt, trinaer_D)
    # trainer.train_G_PG(G=transformer, D=discriminator, optim_G=optim_G, src_seq=src_seq)
    trainer.train(
        0,
        train_iter,
        G_a=transformer_a,
        G_b=transformer_b,
        D_a=discriminator_a,
        D_b=discriminator_b,
        optim_G_a=optim_G_a,
        optim_G_b=optim_G_b,
        optim_D_a=optim_D_a,
        optim_D_b=optim_D_b,
        crit_G=crit_G,
        crit_D=crit_D, )

    # for epoch in range(opt.epoch):
    #     logging.info('[Epoch %d]' % epoch)
    #
    #     train_iter, val_iter = data.BucketIterator.splits(
    #         (train, val), batch_sizes=(opt.batch_size, opt.batch_size), device=opt.device,
    #         sort_key=lambda x: len(x.src), repeat=False)
    #
    #     trainer.train(transformer, train_iter, crit, optimizer, opt)
    #     # trainer.evaluate(transformer, val_iter, crit, EN)
    #
    #     Checkpoint(model=transformer, optimizer=optimizer, epoch=epoch, step=0,
    #                input_vocab=EN.vocab, output_vocab=EN.vocab).save('./experiment/transformer')


if __name__ == '__main__':
    main()
