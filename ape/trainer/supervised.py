from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint


class SupervisedDiscriminatorTrainer(object):
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

    def train(self, model, train_iter, crit, optimizer,
              src_field_name='seq', tgt_field_name='label'):
        model.train()
        step = 0
        for batch in train_iter:
            seq = getattr(batch, src_field_name)
            label = getattr(batch, tgt_field_name)
            batch_size = seq.size(0)

            optimizer.zero_grad()
            prob = model(seq)

            loss = crit(prob, label)
            loss.backward()
            optimizer.step()

            step += batch_size

    def evaluate(self):
        corrects = (prob.gt(0.7).float().view(label.size()).data == label.data).sum()
        accuracy = 100.0 * corrects / batch_size
        self.logger.info('Step[%d] - loss: %.6f acc: %.4f%%(%d/%d)' % (
            step, loss.data[0], accuracy, corrects, batch_size))
