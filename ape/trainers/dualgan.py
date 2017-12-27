import logging
from tqdm import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn

from ape import helper, Constants


class DualGanPGTrainer(object):
    train_D_every = 128  # 每訓練G n次，訓練D 1次

    eval_every = 64
    checkpoint_every = 10000

    n_rollout = 6
    top_k = 3

    lambda_a = 1
    lambda_b = 1

    def __init__(self, opt, trainer_G, trainer_D):
        self.logger = logging.getLogger(__name__)
        self.opt = opt
        self.trainer_G = trainer_G
        self.trainer_D = trainer_D

    def train_G_PG(self, G, D, optim_G, src_seq):
        ''' Policy gradient training on G with beam
        '''
        batch_size = src_seq.size(0)
        for p in D.parameters():
            p.requires_grad = False

        # intermediate D reward
        # Dual training有將還原度一併作為reward，這邊暫時不考慮
        optim_G.zero_grad()

        # encode
        src_pos = G.get_position(src_seq.data)
        enc_output, *_ = G.encoder(src_seq, src_pos)

        # init rollout variable
        # enc_output = helper.stack(enc_output, n_rollout, dim=0)
        # src_seq = helper.stack(src_seq, n_rollout, dim=0)
        cur_seq = autograd.Variable(torch.LongTensor(self.top_k, 1).fill_(Constants.BOS))
        if torch.cuda.is_available():
            cur_seq = cur_seq.cuda()

        rewards, probs = [], []
        final_seqs = []
        candidates = []

        # decode
        for i in range(G.max_len):
            rollouts, sofmax_outs = [], []
            for s in cur_seq.chunk(self.top_k, dim=0):
                rollout_tokens, sofmax_out = G.step_rollout(src_seq, enc_output, s, n_rollout=self.n_rollout)
                rollout_tokens = rollout_tokens.transpose(1, 0)  # (batch * k, 1)
                rollouts.append(rollout_tokens)
                sofmax_outs.append(sofmax_out)

            rollouts = torch.cat(rollouts, dim=0)
            softmax_outs = torch.cat(sofmax_outs, dim=0)

            # 將目前的seq複製成n個，以便與rollout的token(1個seq有n個rollout)結合
            cur_seq = cur_seq.data
            cur_seq = helper.inflate(cur_seq, self.n_rollout, 0)  # (k * n, cur_len)
            cur_seq = torch.cat([cur_seq, rollouts], dim=1)  # (batch * k, cur_len+1)

            _cur_seq = cur_seq.clone()
            if _cur_seq.size(1) < D.min_len:
                _cur_seq = helper.pad_seq(_cur_seq, D.min_len, Constants.PAD)
            reward = D(_cur_seq)  # (batch * k)

            # 儲存rewards, probs，用於計算loss
            # rewards = torch.cat([rewards, reward]) if rewards is not None else reward
            # probs = torch.cat([probs, softmax_outs]) if probs is not None else softmax_outs

            # 從cur_seqs中選出topK的seq
            sorted, indices = reward.sort(dim=0, descending=True)
            candidates = []

            for i in indices.data.split(1):
                seq = cur_seq[i]

                # seq是否存在candidates中？ 若沒有則加入candidates
                if not any(torch.equal(seq, x) for x in candidates):
                    # 若candidate的最新一個token為EOS，則加入final_seqs
                    if seq[:, -1][0] == Constants.EOS:
                        final_seqs.append(seq)
                    else:
                        candidates.append(seq)

                    # 儲存被選上的rewards, probs
                    rewards.append(reward[i])
                    probs.append(softmax_outs[i])

                    if len(candidates) == (self.top_k - len(final_seqs)):
                        break

            # 判斷beams皆已完成？
            if len(candidates) == 0:
                break
            else:
                cur_seq = autograd.Variable(torch.cat(candidates, dim=0))

        final_seqs += candidates
        rewards = torch.cat(rewards)
        probs = torch.cat(probs)

        # print(rewards)
        # print(probs)

        # back propagation
        loss = -torch.mean(rewards * probs)
        loss.backward()
        nn.utils.clip_grad_norm(G.get_trainable_parameters(), 40)  # 避免grad爆炸
        optim_G.step()

        return final_seqs[0], rewards, probs, loss

    def _train_G_PG(self, G, D, optim_G, src_seq):
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

        # decode
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
            next_token = autograd.Variable(next_token.unsqueeze(1))  # 需轉為variable，torch.cat()才不會出錯

            dec_seq = torch.cat([dec_seq, next_token], dim=1)

            rewards = torch.cat([rewards, reward]) if rewards is not None else reward
            probs = torch.cat([probs, prob]) if probs is not None else reward
            # probs += list(prob.split(1))

            if next_token[0] == Constants.EOS:
                break

        # back propagation
        # print(probs)
        # print(rewards)

        loss = -torch.mean(rewards * probs)

        # print(loss)

        loss.backward()
        nn.utils.clip_grad_norm(G.get_trainable_parameters(), 40)  # 避免grad爆炸
        optim_G.step()

        return dec_seq, rewards, probs, loss

    def train_D(self, train_iter, D, G, crit_D, optim_D, n_step=256):
        for p in G.parameters():
            p.requires_grad = False

        pool = helper.DiscriminatorDataPool(G.n_max_seq, D.min_len, Constants.PAD)
        for i, batch in enumerate(train_iter):
            src_seq = batch.src
            _, hyp_seq = G.translate(src_seq)
            pool.append_fake(hyp_seq, to_partial=True)
            pool.append_real(batch.tgt, to_partial=True)
            if i > n_step:
                break
        self.trainer_D.train(D, train_iter=pool.batch_gen(),
                             crit=crit_D, optimizer=optim_D)

        for p in G.parameters():
            p.requires_grad = True

    def train(self, epoch, train_iter, G_a, G_b, D_a, D_b,
              optim_G_a, optim_G_b, optim_D_a, optim_D_b, crit_G, crit_D,
              eval_G, A_FIELD, B_FIELD):
        # pretrain D
        # self.train_D(train_iter, D_a, G_b, crit_D, optim_D_a)
        # self.train_D(train_iter, D_b, G_a, crit_D, optim_D_a)

        D_a.train()
        D_b.train()
        G_a.train()
        G_b.train()

        pool_a = helper.DiscriminatorDataPool(G_a.n_max_seq, D_a.min_len, Constants.PAD)
        pool_b = helper.DiscriminatorDataPool(G_b.n_max_seq, D_b.min_len, Constants.PAD)

        samples_a2b, samples_b2a = [], []

        # step = len(train_iter) * epoch
        step = 0
        for batch in tqdm(train_iter):
            real_a = batch.src
            real_b = batch.tgt
            batch_size = real_a.size(0)

            # (1) train G
            # (1.1) realness of G by policy gradient
            # fake_b, rewards_G_a, probs_G_a, loss_pg_G_a = self.train_G_PG(G_a, D_b, optim_G_a, src_seq=real_a)
            # fake_a, rewards_G_b, probs_G_b, loss_pg_G_b = self.train_G_PG(G_b, D_a, optim_G_b, src_seq=real_b)
            # samples_a2b.append((real_a, fake_b))
            # samples_b2a.append((real_b, fake_a))

            # (1.2) cycle loss
            optim_G_a.zero_grad()
            optim_G_b.zero_grad()

            logit, fake_b = G_a.translate(real_a)
            # _fake_b = G_a.decode_seq(logit)
            rest_a_logit = G_b(fake_b, real_a)
            _rest_a = G_a.decode_seq(rest_a_logit)

            logit, fake_a = G_b.translate(real_b)
            # _fake_a = G_b.decode_seq(logit)
            rest_b_logit = G_a(fake_a, real_b)
            _rest_b = G_b.decode_seq(rest_b_logit)

            # fake_a = autograd.Variable(fake_a)
            # fake_b = autograd.Variable(fake_b)
            # if self.opt.cuda:
            #     fake_a.cuda()
            #     fake_b.cuda()

            # logit = G_a(real_a, fake_b)  # 重跑一遍獲得gradients
            # _fake_b = G_a.decode_seq(logit)
            # rest_a_logit = G_b(_fake_b, real_a)  # 使用teacher-forcing
            # # rest_a = G_b.translate(fake_b)  # 不使用teacher-forcing
            #
            # logit = G_b(real_b, fake_a)
            # _fake_a = G_b.decode_seq(logit)
            # rest_b_logit = G_a(_fake_a, real_b)

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


            # pool_a.append_fake(fake_a, to_partial=True)
            # pool_a.append_real(real_a, to_partial=True)
            # pool_b.append_fake(fake_b, to_partial=True)
            # pool_b.append_real(real_b, to_partial=True)

            # (2) train D
            if step % self.train_D_every == 0:
                # for p in D_a.parameters():
                #     p.requires_grad = True
                # for p in D_b.parameters():
                #     p.requires_grad = True
                # self.logger.info('[Step %d] Train D_a', step)
                # self.trainer_D.train(D_a,
                #                      train_iter=pool_a.batch_gen(),
                #                      crit=crit_D,
                #                      optimizer=optim_D_a)
                # self.logger.info('[Step %d] Train D_b', step)
                # self.trainer_D.train(D_b,
                #                      train_iter=pool_b.batch_gen(),
                #                      crit=crit_D,
                #                      optimizer=optim_D_b)
                pool_a.reset()
                pool_b.reset()



            # logging & checkpoint
            if step % self.eval_every == 0:
                # self.logger.info('[PG train] G_a: rewards %.4f(%.4f), pg_loss %.4F' % (
                #     rewards_G_a.mean().data[0], rewards_G_a.var().data[0], loss_pg_G_a.data[0]))
                # self.logger.info('[PG train] G_b: rewards %.4f(%.4f), pg_loss %.4F' % (
                #     rewards_G_b.mean().data[0], rewards_G_b.var().data[0], loss_pg_G_b.data[0]))
                # self.logger.info('[Cycle train] cycle_loss_a %.4f, cycle_loss_b %.4f' % (
                #     loss_cycle_a.data[0], loss_cycle_b.data[0]))

                # self.logger.info('\n--------  real_a -> fake_b  --------')
                # for a, b in samples_a2b[:10]:
                #     self.logger.info('%s -> %s' % (A_FIELD.reverse(a)[0], B_FIELD.reverse(b)[0]))
                # self.logger.info('--------  real_b -> fake_a  --------')
                # for b, a in samples_b2a[:10]:
                #     self.logger.info('%s -> %s' % (B_FIELD.reverse(b)[0], A_FIELD.reverse(a)[0]))
                # samples_a2b, samples_b2a = [], []
                # self.logger.info('\n')

                self.logger.info('[Evaluate G_a]')
                eval_G(G_a)
                # self.logger.info('[Evaluate G_b]')
                # eval_G(G_b)

                self.logger.info('Loss %.4f' % loss_cycle)
                self.logger.info('\n')
                self.logger.info('[A->B->A]')
                self.logger.info('%s -> %s -> %s' % (
                    A_FIELD.reverse(real_a)[0], B_FIELD.reverse(fake_b)[0], A_FIELD.reverse(_rest_a)[0]))
                self.logger.info('[B->A->B]')
                self.logger.info('%s -> %s -> %s\n' % (
                    B_FIELD.reverse(real_b)[0], A_FIELD.reverse(fake_a)[0], B_FIELD.reverse(_rest_b)[0]))
            if step % self.checkpoint_every == 0:
                pass
            step += 1
