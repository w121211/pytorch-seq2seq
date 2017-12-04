from __future__ import division
import time
import logging
import numpy as np
from tqdm import tqdm

import torch
# import torch.nn as nn
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torchtext

import seq2seq
from seq2seq.models.gan import ReinforceGenerator
from seq2seq.evaluator import bleu
from seq2seq.util import helper
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.models.transformer import Constants
from .supervised_trainer import SupervisedTrainer


class GanTrainer(SupervisedTrainer):
    '''
    和原本的SupervisedTrainer比較，BaseGanTrainer增加：evaluate(), predict()。
    設計概念為trainer負責做run dataset的相關任務，因此不管是train, evaluate, predict都是類似的性質，因此放在同一個class下。
    '''
    device = None if torch.cuda.is_available() else -1

    def __init__(self, src_field_name='src', tgt_field_name='tgt', **kwargs):
        super(GanTrainer, self).__init__(**kwargs)
        self.src_field_name = src_field_name
        self.tgt_field_name = tgt_field_name

    def train(self, model, dis, real_data, num_rollout=64,
              dev_data=None, optimizer=None):
        '''
        用 policy gradient 訓練 seq2seq model，步驟：
            1. 輸入一個source seq，encoder把它encode
            2. 將 encoded seq 給decoder，decoder依序產生下個字的機率表，依'該機率'決定下個字 x(decode)，
               並將 x 作為input，繼續產生下一個字。
            3. 持續步驟2，直至生成n個samples
            4. Evaluator將每個samples給reward，並以此計算中間的各個action的reward
            5. 以此rewards做back propagation˙
        '''
        model.train()

        real_iter = torchtext.data.BucketIterator(
            (real_data,),
            batch_size=(1,))

        step = 0
        self.logger.info('[Step %d]Start policy gradient training...' % step)
        for batch in real_iter:
            input_variable, input_lengths = getattr(batch, seq2seq.src_field_name)
            loss = self.pg_loss(model, dis, input_variable, input_lengths, num_rollout)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.logger.info('[Step %d]PG_loss % .4f' % (step, loss.data[0]))
            step += 1
        self.logger.info('[Step %d]Finish policy gradient training')

    def pretrain_G(self, model, train, criterion, optimizer, num_epoch=20, val=None,
                   FIELD_TGT=None):
        if not isinstance(model, ReinforceGenerator):
            raise TypeError('model type 需為 %s' % type(ReinforceGenerator))

        self.logger.info('Start training generator...')

        for epoch in range(num_epoch):
            model.train()
            train_iter, val_iter = torchtext.data.BucketIterator.splits(
                (train, val), batch_sizes=(64, len(val)), device=self.device,
                sort_key=lambda x: len(getattr(x, seq2seq.src_field_name)), repeat=False)

            step = 0
            for batch in tqdm(train_iter, mininterval=2,
                              desc='  - (Training)   ',
                              leave=False):
                src, src_length = getattr(batch, seq2seq.src_field_name)
                tgt, tgt_length = getattr(batch, seq2seq.tgt_field_name)
                batch_size = src.size(0)

                optimizer.zero_grad()
                hyp, hyp_length, hyp_probs = model(src, src_length.tolist())
                for loss in model.batch_loss(hyp_probs, tgt, criterion=criterion):
                    loss.backward(retain_graph=True)
                    optimizer.step()
                # loss.backward()

                step += batch_size
                if step % self.print_every == 0:
                    self.logger.info('[Epoch %d] %s: %.4f (%d/%d)' % (
                        epoch, criterion.__class__.__name__, loss.data[0], step, len(train_iter) * batch_size))
                    # break
            # break
            # evaluation after each epoch
            self.evaluate_G(model, val_iter, criterion, FIELD_TGT)

    def evaluate_G(self, model, val_iter, criterion, FIELD_TGT):
        model.eval()

        loss = 0
        tgt, hyp = None, None
        tgts, hyps = [], []
        for batch in val_iter:
            src, src_length = getattr(batch, seq2seq.src_field_name)
            tgt, tgt_length = getattr(batch, seq2seq.tgt_field_name)

            hyp, hyp_length, hyp_probs = model(src, src_length.tolist())
            loss += model.batch_loss(hyp_probs, tgt, criterion=criterion)

            tgts += FIELD_TGT.reverse(tgt.data)
            hyps += FIELD_TGT.reverse(hyp.data)
        _bleu = bleu.moses_multi_bleu(hypotheses=np.array(hyps), references=np.array(tgts),
                                      lowercase=True)

        print(tgt)
        print(hyp)

        logging.info("[Evaluate] %s: %.4f, BLEU: %.2f" % (criterion.__class__.__name__, loss.data[0], _bleu))
        for ref, hyp in zip(tgts[0:5], hyps[0:5]):
            logging.info('(%s) || (%s)' % (ref, hyp))

    def pretrain_D(self, D, G, train, criterion, optimizer, num_epoch=1,
                   src_field_name=None, tgt_field_name=None):
        src_name = self.src_field_name if src_field_name is None else src_field_name
        tgt_name = self.tgt_field_name if tgt_field_name is None else tgt_field_name

        for epoch in range(num_epoch):
            D.train()

            train_iter, = torchtext.data.BucketIterator.splits(
                (train,), batch_sizes=(1,), device=self.device,
                sort_key=lambda x: len(x.real_a), repeat=False)

            pool = helper.DDataPool(max_len=G.max_len)
            for batch in train_iter:
                src, src_length = getattr(batch, src_name)
                tgt, tgt_length = getattr(batch, tgt_name)
                rollout, _, _, _ = G.rollout(src, num_rollout=1)
                pool.append_fake(rollout[0, :].contiguous().view(1, -1))
                pool.append_real(G._validate_variables(target=tgt).data)

                if len(pool.fakes) > 1000:
                    break
            self.supervised_train_D(D, pool.batch_gen(), criterion, optimizer)

    def supervised_train_D(self, model, train_iter, criterion, optimizer,
                           src_field_name='seq', tgt_field_name='label'):
        model.train()
        step = 0
        for batch in train_iter:
            seq = getattr(batch, src_field_name)
            label = getattr(batch, tgt_field_name)
            batch_size = seq.size(0)

            optimizer.zero_grad()
            prob = model(seq)
            loss = criterion(prob, label)
            loss.backward()
            optimizer.step()

            # if step % 64 == 0:
            #     corrects = (prob.gt(0.7).float().view(label.size()).data == label.data).sum()
            #     accuracy = 100.0 * corrects / batch_size
            #     self.logger.info('Step[%d] - loss: %.6f acc: %.4f%%(%d/%d)' % (
            #         step, loss.data[0], accuracy, corrects, batch_size))
            step += batch_size

    def predict(self, *args, **kwargs):
        raise NotImplementedError()


class CycleGanReinforceTrainer(GanTrainer):
    every_train_D = 128  # 每train_G 128次，train_D 1次

    def __init__(self, model, A_FIELD, B_FIELD, lambda_a=1., lambda_b=1., **kwargs):
        super(CycleGanReinforceTrainer, self).__init__(**kwargs)
        del self.evaluator  # 原始的evaluator在這裡定義不明，所以刪除
        self.model = model
        self.criterion_G = nn.NLLLoss()
        # self.optimizer_G = torch.optim.Adam(
        #     list(self.model.g_a.parameters()) + list(self.model.g_b.parameters()), lr=1e-2)
        self.optimizer_G = torch.optim.SGD(
            list(self.model.g_a.parameters()) + list(self.model.g_b.parameters()), lr=1e-2)
        self.criterion_D = nn.BCELoss()
        self.optimizer_D_a = torch.optim.Adam(self.model.d_a.parameters(), lr=1e-2)
        self.optimizer_D_b = torch.optim.Adam(self.model.d_b.parameters(), lr=1e-2)
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.A_FIELD = A_FIELD
        self.B_FIELD = B_FIELD

        self.logger = logging.getLogger(__name__)

    def train(self, train, val=None, num_epoch=200, resume=False):
        start_epoch = 0
        if resume:
            cp = Checkpoint.load(Checkpoint.get_latest_checkpoint('./experiment/gan'))
            self.model = cp.model
            start_epoch = cp.epoch + 1

        for epoch in range(start_epoch, num_epoch):
            logging.info('Epoch[%d] CycleGAN train' % epoch)

            train_iter, val_iter = torchtext.data.BucketIterator.splits(
                (train, val), batch_sizes=(1, 64), device=self.device,
                sort_key=lambda x: len(x.real_a), repeat=False)

            self._train_epoch(train_iter)
            self.evaluate(val_iter)

            Checkpoint(model=self.model, optimizer=None, epoch=epoch, step=0,
                       input_vocab=None, output_vocab=None).save('./experiment/gan')

    def _train_epoch(self, train_iter):
        self.model.train()

        pool_a = helper.DDataPool(self.model.g_a.max_len)
        pool_b = helper.DDataPool(self.model.g_a.max_len)
        step = 0
        for batch in train_iter:
            real_a, real_a_length = getattr(batch, self.src_field_name)
            real_b, real_b_length = getattr(batch, self.tgt_field_name)
            batch_size = real_a.size(0)

            # (1) train G
            # (1.1) Loss for restoration
            # TODO 移除input的<sos>?
            self.optimizer_G.zero_grad()

            fake_b, fake_b_length, fake_b_probs = self.model.g_a(real_a, real_a_length.tolist())
            rest_a, _, rest_a_probs = self.model.g_b(fake_b, None)  # restored a
            fake_a, fake_a_length, fake_a_probs = self.model.g_b(real_b, real_b_length.tolist())
            rest_b, _, rest_b_probs = self.model.g_a(fake_a, None)  # restored b

            loss_cycle_a = self.model.g_a.batch_loss(rest_a_probs, real_a, criterion=self.criterion_G)
            loss_cycle_b = self.model.g_b.batch_loss(rest_b_probs, real_b, criterion=self.criterion_G)

            # (1.2) Loss for GAN
            rollouts_b, seq_symbols, log_probs, entropies = self.model.g_a.rollout(real_a, num_rollout=64)
            rewards_a = self.model.d_b(autograd.Variable(rollouts_b))
            loss_pg_a = self.model.pg_loss(log_probs, entropies, rewards_a)  # policy gradient loss

            rollouts_a, seq_symbols, log_probs, entropies = self.model.g_b.rollout(real_b, num_rollout=64)
            rewards_b = self.model.d_a(autograd.Variable(rollouts_a))
            loss_pg_b = self.model.pg_loss(log_probs, entropies, rewards_b)  # policy gradient loss

            loss_G = self.lambda_a * loss_cycle_a + self.lambda_b * loss_cycle_b + loss_pg_a + loss_pg_b
            loss_G.backward()
            self.optimizer_G.step()

            pool_a.append_fake(rollouts_a[0, :].contiguous().view(1, -1))
            pool_a.append_real(self.model.g_a._validate_variables(target=real_a).data)
            pool_b.append_fake(rollouts_b[0, :].contiguous().view(1, -1))
            pool_b.append_real(self.model.g_b._validate_variables(target=real_b).data)

            # (2) train D
            if step % self.every_train_D == 0:
                self.logger.info('Step[%d] Train D', step)
                self.supervised_train_D(self.model.d_a, train_iter=pool_a.batch_gen(),
                                        criterion=self.criterion_D, optimizer=self.optimizer_D_a)
                self.supervised_train_D(self.model.d_b, train_iter=pool_b.batch_gen(),
                                        criterion=self.criterion_D, optimizer=self.optimizer_D_b)
                self.logger.info('rewards_a: avg %.4f, var %.4f' % (rewards_a.mean().data[0], rewards_a.var().data[0]))
                self.logger.info('rewards_b: avg %.4f, var %.4f' % (rewards_b.mean().data[0], rewards_b.var().data[0]))
                self.logger.info('loss_pg_a: %.4f' % loss_pg_a.data[0])
                self.logger.info('loss_pg_b: %.4f' % loss_pg_b.data[0])
                pool_a.reset()
                pool_b.reset()
            if step % self.print_every == 0:
                self.logger.info('loss g: %.4f (%d/%d)' % (loss_G.data[0], step, len(train_iter) * batch_size))
            step += batch_size

    def evaluate(self, val_iter):
        self.model.g_a.eval()
        self.model.g_b.eval()

        loss_G_a, loss_G_b = 0, 0
        reals_a, reals_b, fakes_a, fakes_b = [], [], [], []
        real_a, fake_a, real_b, fake_b = None, None, None, None
        for batch in val_iter:
            real_a, real_a_length = batch.real_a
            real_b, real_b_length = batch.real_b  # 此為real_a轉換為real_b的正解

            fake_b, _, fake_b_probs = self.model.g_a(real_a, None)
            fake_a, _, fake_a_probs = self.model.g_b(real_b, None)

            loss_G_a += self.model.g_a.batch_loss(fake_b_probs, real_b, criterion=self.criterion_G)
            loss_G_b += self.model.g_b.batch_loss(fake_a_probs, real_a, criterion=self.criterion_G)

            reals_a += self.A_FIELD.reverse(real_a.data)
            reals_b += self.B_FIELD.reverse(real_b.data)
            fakes_a += self.A_FIELD.reverse(fake_a.data)
            fakes_b += self.B_FIELD.reverse(fake_b.data)

        bleu_G_a = bleu.moses_multi_bleu(hypotheses=np.array(fakes_b), references=np.array(reals_b),
                                         lowercase=True)
        bleu_G_b = bleu.moses_multi_bleu(hypotheses=np.array(fakes_a), references=np.array(reals_a),
                                         lowercase=True)

        # loss_g_a /= len(val_iter)

        print(real_a)
        print(fake_a)
        print('---------')
        print(real_b)
        print(fake_b)

        logging.info("[Val(%d) G_a]  %s: %.4f, BLEU: %.2f" % (
            len(val_iter), self.criterion_G.__class__.__name__, loss_G_a.data[0], bleu_G_a))
        logging.info("[Val(%d) G_b]  %s: %.4f, BLEU: %.2f" % (
            len(val_iter), self.criterion_G.__class__.__name__, loss_G_b.data[0], bleu_G_b))
        for ref, hyp in zip(reals_b[0:5], fakes_b[0:5]):
            logging.info('(%s) || (%s)' % (ref, hyp))

    def predict(self, test):
        pass


class WganTrainer(object):
    '''
    Adopt WGAN-GP & CycleGAN methods for training
    '''

    # every_train_D = 128  # 每train_G 128次，train_D 1次

    CRITIC_ITERS = 5  # How many critic iterations per generator iteration
    LAMBDA = 10

    def __init__(self, opt):
        self.opt = opt

    def gradient_penalty(self, D, real, fake):
        batch_size = real.size(0)

        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand(real.size())
        if self.opt.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real + ((1 - alpha) * fake)
        if self.opt.cuda:
            interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = D(interpolates)

        # TODO: Make ConvBackward diffentiable
        grad_outputs = torch.ones(disc_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(
            disc_interpolates.size())
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA

        return gradient_penalty

    def train(self, D, G, optimizer_D, optimizer_G,
              train, val=None,
              num_epoch=200, resume=False, opt=None):
        start_epoch = 0
        if resume:
            cp = Checkpoint.load(Checkpoint.get_latest_checkpoint('./experiment/gan'))
            self.model = cp.model
            start_epoch = cp.epoch + 1

        for epoch in range(start_epoch, num_epoch):
            logging.info('Epoch[%d] CycleGAN train' % epoch)

            train_iter, val_iter = torchtext.data.BucketIterator.splits(
                (train, val), batch_sizes=(1, 64), device=opt.device,
                sort_key=lambda x: len(x.real_a), repeat=False)

            self.train_epoch(D, G, optimizer_D, optimizer_G, train_iter)
            # self.evaluate(val_iter)

            # Checkpoint(model=self.model, optimizer=None, epoch=epoch, step=0,
            #            input_vocab=None, output_vocab=None).save('./experiment/gan')

    def train_epoch(self, epoch, D, G, optimizer_D, optimizer_G,
                    train_iter, n_tgt_vocab):
        D.train()
        G.train()

        one = torch.FloatTensor([1])
        mone = one * -1
        if torch.cuda.is_available():
            one = one.cuda()
            mone = mone.cuda()

        step = epoch * len(train_iter)
        for batch in train_iter:
            src_seq = batch.src
            src_pos = G.get_position(src_seq.data)

            real = batch.tgt.cpu().data
            real = real[:, 1:]  # 移除<bos>
            real = helper.pad_sequence(real, max_len=D.max_len, pad_value=Constants.PAD)
            real = torch.FloatTensor(
                real.size(0), real.size(1), n_tgt_vocab).zero_().scatter_(2, real.unsqueeze(2), 1)  # 轉成 one-hot
            real = autograd.Variable(real)
            if torch.cuda.is_available():
                real = real.cuda()

            # (1) train G
            for p in D.parameters():
                p.requires_grad = False

            optimizer_G.zero_grad()

            fake = G.translate(src_seq, src_pos)

            D_fake = D(fake)
            D_fake = D_fake.mean()
            D_fake.backward(mone)
            optimizer_G.step()

            cost_G = -D_fake.cpu().data.numpy()

            # (2) train D
            for p_d, p_g in zip(D.parameters(), G.parameters()):
                p_d.requires_grad = True
                p_g.requires_grad = False

            optimizer_D.zero_grad()

            # print('------')
            # print(fake)
            # print(real)

            D_real = D(real)
            D_real = D_real.mean()
            D_real.backward(mone)

            fake = G.translate(src_seq, src_pos)
            D_fake = D(fake)
            D_fake = D_fake.mean()

            gp = self.gradient_penalty(D, real.data, fake.data)
            gp.backward()
            optimizer_D.step()

            cost_D = (D_fake - D_real + gp).cpu().data.numpy()
            wasserstein_D = (D_real - D_fake).cpu().data.numpy()

            print('[step %d] cost_G: %.4f, cost_D: %.4f, Wasserstein_D: %.4f' % (
                step, cost_G, cost_D, wasserstein_D))

            step += 1
