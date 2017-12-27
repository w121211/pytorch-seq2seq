import logging
from tqdm import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn

from ape import helper, Constants


class CycleTrainer(object):
    eval_every = 64
    checkpoint_every = 10000

    lambda_a = 1
    lambda_b = 1

    def __init__(self, opt, trainer_G, loss):
        self.logger = logging.getLogger(__name__)
        self.opt = opt
        self.trainer_G = trainer_G

        self.loss = loss

    def train(self, train_iter, G_a, G_b, optim_G_a, optim_G_b, crit_G, A_FIELD, B_FIELD):
        G_a.train()
        G_b.train()

        for step, batch in enumerate(tqdm(train_iter)):
            real_a = batch.src
            real_b = batch.tgt
            # batch_size = real_a.size(0)

            optim_G_a.zero_grad()
            optim_G_b.zero_grad()

            decoder_outputs, decoder_hidden, other = G_a(
                real_a[0], real_a[1].tolist(), None, teacher_forcing_ratio=0)
            length = [self.opt.max_len for _ in other['length']]
            fake_b = torch.cat([other['sequence'][di] for di in range(self.opt.max_len)], dim=1)
            # print(length)
            # print(fake_b)

            loss = self.loss
            decoder_outputs, decoder_hidden, other = G_b(
                fake_b, length, real_a[0], teacher_forcing_ratio=0)

            loss.reset()
            for _step, step_output in enumerate(decoder_outputs):
                batch_size = real_a[0].size(0)
                loss.eval_batch(step_output.contiguous().view(batch_size, -1), real_a[0][:, _step + 1])

            loss.backward()
            optim_G_a.step()
            optim_G_b.step()

            length = other['length'][0]
            rest_a = torch.LongTensor([other['sequence'][di][0].data[0] for di in range(length)]).unsqueeze(0)
            # print(rest_a)

            if step % self.eval_every == 0:
                # self.logger.info('[Evaluate G_a]')
                # eval_G(G_a)
                # self.logger.info('[Evaluate G_b]')
                # eval_G(G_b)

                self.logger.info('\nppl %.4f' % loss.get_loss())
                self.logger.info('[A->B->A]')
                self.logger.info('%s -> %s -> %s' % (
                    A_FIELD.reverse(real_a[0].data)[0],
                    B_FIELD.reverse(fake_b.data)[0],
                    A_FIELD.reverse(rest_a)[0]))
                # self.logger.info('[B->A->B]')
                # self.logger.info('%s -> %s -> %s\n' % (
                #     B_FIELD.reverse(real_b)[0], A_FIELD.reverse(fake_a)[0], B_FIELD.reverse(_rest_b)[0]))
                self.logger.info('\n')
            if step % self.checkpoint_every == 0:
                pass

    def _train(self, train_iter, G_a, G_b, optim_G_a, optim_G_b, crit_G, A_FIELD, B_FIELD):
        G_a.train()
        G_b.train()

        for step, batch in enumerate(tqdm(train_iter)):
            real_a = batch.src
            real_b = batch.tgt
            batch_size = real_a.size(0)

            optim_G_a.zero_grad()
            optim_G_b.zero_grad()

            fake_b = G_a.translate(real_a)
            rest_a_logit = G_b(fake_b, real_a)
            rest_a = G_a.decode_seq(rest_a_logit)

            def criterion(crit, pred, gold):
                gold = gold[:, 1:]  # 去掉<bos>
                pred = pred.view(-1, pred.size(2))  # (batch*len, n_vocab)
                return crit(pred, gold.contiguous().view(-1))

            loss = criterion(crit_G, rest_a_logit, real_a)

            # loss_cycle = self.lambda_a * loss_cycle_a + self.lambda_b * loss_cycle_b
            loss.backward()

            optim_G_a.step()
            optim_G_b.step()

            if step % self.eval_every == 0:
                # self.logger.info('[Evaluate G_a]')
                # eval_G(G_a)
                # self.logger.info('[Evaluate G_b]')
                # eval_G(G_b)

                # self.logger.info('\nppl %.4f' % loss.data[0])
                self.logger.info('\nppl %.4f' % loss.get_loss())
                self.logger.info('[A->B->A]')
                self.logger.info('%s -> %s -> %s' % (
                    A_FIELD.reverse(real_a.data)[0],
                    B_FIELD.reverse(fake_b.data)[0],
                    A_FIELD.reverse(rest_a.data)[0]))
                # self.logger.info('[B->A->B]')
                # self.logger.info('%s -> %s -> %s\n' % (
                #     B_FIELD.reverse(real_b)[0], A_FIELD.reverse(fake_a)[0], B_FIELD.reverse(_rest_b)[0]))
                self.logger.info('\n')
            if step % self.checkpoint_every == 0:
                pass
