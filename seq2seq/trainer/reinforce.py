from __future__ import division
import logging
import os
import random
import time

import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torchtext
from torch import optim

import seq2seq

from .supervised_trainer import SupervisedTrainer


class PolicyGradientTrainer(SupervisedTrainer):
    def __init__(self, max_len, **kwargs):
        super(PolicyGradientTrainer, self).__init__(**kwargs)
        self.max_len = max_len
        self.logger = logging.getLogger(__name__)

    def _sample(self, model, encoder_hidden, encoder_outputs, num_sample):
        '''

        :param model:
        :param encoder_hidden: (h_n, c_n)
        :param encoder_outputs: h_n
        :return:
        '''
        # decoder_input = inputs[:, 0].unsqueeze(1)
        # samples = torch.zeros(
        #     self.num_sample, self.max_len).type(torch.LongTensor)  # 存sample sequence
        # sample_probs = torch.zeros(
        #     self.num_sample, self.max_len).type(torch.DoubleTensor)  # 存所選action的prob
        actions = []  # a list of variables (size: num_samples x 1)
        log_probs = []
        entropies = []

        inp = autograd.Variable(
            torch.LongTensor([model.decoder.sos_id] * num_sample).view(num_sample, 1))  # num_sample x 1
        if torch.cuda.is_available():
            # samples = samples.cuda()
            inp = inp.cuda()

        decoder_input = inp
        h_n, c_n = model.decoder._init_state(encoder_hidden)
        h_n = torch.cat([h_n for _ in range(num_sample)])  # 複製數個用於batch
        c_n = torch.cat([c_n for _ in range(num_sample)])  # 複製數個用於batch
        hidden = (h_n, c_n)
        encoder_outputs = torch.cat([encoder_outputs for _ in range(num_sample)])  # 複製數個用於batch

        _indices = list(range(num_sample))
        for i in range(self.max_len - 1):
            softmax, hidden, attn = model.decoder.forward_step(
                decoder_input, hidden, encoder_outputs, F.softmax)
            batch_probs = softmax.squeeze(1)
            # print(batch_probs)

            batch_action = batch_probs.multinomial(1).data
            actions.append(batch_action)

            batch_prob = batch_probs[_indices, batch_action.squeeze(1).tolist()]
            # print(batch_prob.log())
            log_probs.append(batch_prob.log())

            batch_entropy = -(batch_probs * batch_probs.log()).sum(dim=1)
            entropies.append(batch_entropy)
            # break

        # actions = []  # a list of variables (size: 1x1)
        # decodes = []  # a list of (vocab_id, prob_of_vocab)
        #
        # decoder_input = inputs[:, 0].unsqueeze(1)
        # hidden = model.decoder._init_state(encoder_hidden)
        # for i in range(self.max_len):
        #     probs, hidden, attn = model.decoder.forward_step(
        #         decoder_input, hidden, encoder_outputs, F.log_softmax)
        #
        #     # decode
        #     probs = probs.exp().squeeze(0)
        #     action = probs.multinomial(1)  # Variable (size: 1x1), 從vocab_probs中隨機選出一個vocab_id
        #     symbol = action.data[0, 0]  # 取得實際的vocab_id
        #     actions.append(action)
        #     decodes.append((symbol, probs.data[0, symbol]))  # (vocab_id, prob_vocab)
        #     decoder_input = action
        #     if symbol == model.decoder.eos_id:
        #         break
        return actions, log_probs, entropies

    def _batch_loss(self):
        pass

    def train(self, model, rewarder, train_iter,
              num_src=100, src2sample=64,
              resume=False, dev_data=None, optimizer=None):
        '''用 policy gradient 訓練 seq2seq model，步驟：
            1. 輸入一個source seq，encoder把它encode
            2. 將 encoded seq 給decoder，decoder依序產生下個字的機率表，依'該機率'決定下個字 x(decode)，
            並將 x 作為input，繼續產生下一個字。
            3. 持續步驟2，直至生成n個samples
            4. Evaluator將每個samples給reward，並以此計算中間的各個action的reward
            5. 以此rewards做back propagation
        '''
        step = 0
        loss = None
        for batch_sample, actions, log_probs, entropies in self.gen_sample(
                model, train_iter, num_src, src2sample):

            # 先算整體reward，再計算每個action的reward
            # print(batch_sample)
            rewards = rewarder(autograd.Variable(batch_sample))
            print(rewards.mean())
            # rewards = autograd.Variable(torch.Tensor([10] * self.num_sample))
            if torch.cuda.is_available():
                rewards = rewards.cuda()
            # break
            loss = 0
            for log_prob, entropy in zip(log_probs, entropies):
                # print(log_prob * rewards)
                loss += -(log_prob * rewards).sum()
                # loss += -((log_prob * rewards).sum() + (0.0001 * entropy).sum())
                # action.reinforce(reward)
                # print(loss)

            loss /= len(log_probs) * src2sample
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # self.logger.info('loss % .4f' % loss.data[0])

            del actions
            del log_probs
            del entropies

            step += 1

        # evaluate
        self.logger.info('Finish epoch of reinforce training')
        if dev_data is not None:
            dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
            self.logger.info('Dev %s: %.4f, Accuracy: %.4f' % (self.loss.name, dev_loss, accuracy))
            model.train(mode=True)

    def gen_sample(self, model, batch_iter, num_src, src2sample=32):
        self.logger.info('Generate samples ...')
        for step, batch in enumerate(batch_iter):
            if step >= num_src:
                break

            input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
            if input_variables.size(0) != 1:
                raise Exception('Policy gradient trainer只接受 batch_size = 1')

            encoder_outputs, encoder_hidden = model.encoder(input_variables, input_lengths.tolist())
            actions, log_probs, entropies = self._sample(
                model, encoder_hidden, encoder_outputs, src2sample)
            batch_sample = torch.cat(actions, 1)
            yield batch_sample, actions, log_probs, entropies
