from __future__ import division
import logging
import math

import numpy as np
from tqdm import tqdm

from seq2seq.util.checkpoint import Checkpoint

from ape import Constants, helper
from ape.evaluator import bleu as BLEU


class TransformerTrainer(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_performance(self, crit, pred, gold, smoothing=False, num_class=None):
        ''' Apply label smoothing if needed '''

        # TODO: Add smoothing
        if smoothing:
            assert bool(num_class)
            eps = 0.1
            gold = gold * (1 - eps) + (1 - gold) * eps / num_class
            raise NotImplementedError

        pred = pred.view(-1, pred.size(2))  # (batch*len, n_vocab)
        loss = crit(pred, gold.contiguous().view(-1))

        pred = pred.max(1)[1]

        gold = gold.contiguous().view(-1)
        n_correct = pred.data.eq(gold.data)
        n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum()

        return loss, n_correct
        # return n_correct

    def evaluate(self, model, val_iter, crit, TGT_FIELD):
        model.eval()

        total_loss = 0
        n_total_words = 0
        n_total_correct = 0
        hyps = []
        reals = []

        for batch in tqdm(val_iter, mininterval=2,
                          desc='(Validation) ', leave=False):
            # prepare data
            src_seq = batch.src
            tgt_seq = batch.tgt
            gold = tgt_seq[:, 1:]

            # forward
            src_seq.volatile = True
            logit = model(src_seq, tgt_seq)
            # dec_output = model(src_seq, src_pos, tgt_seq, tgt_pos)
            # loss = model.batch_loss(crit, dec_output, gold)
            loss, n_correct = self.get_performance(crit, pred=logit, gold=gold)

            # dec = autograd.Variable(dec_output.data, volatile=True)
            # logit = model.tgt_word_proj(dec)
            # logit = logit.view(-1, logit.size(2))
            # n_correct = self.get_performance(crit, pred=logit, gold=gold)

            # note keeping
            n_words = gold.data.ne(Constants.PAD).sum()
            n_total_words += n_words
            n_total_correct += n_correct
            total_loss += loss.data[0]

            pred = logit.max(2)[1]
            pred = helper.pad_bos(pred.data, Constants.BOS)

            hyps += TGT_FIELD.reverse(pred)
            reals += TGT_FIELD.reverse(tgt_seq.data)

        # bleu = BLEU.moses_multi_bleu(hypotheses=np.array(hyps),
        #                              references=np.array(reals),
        #                              lowercase=True)
        bleu = -float('Inf')

        self.logger.info('\n')
        for real, hyp in zip(reals[0:10], hyps[0:10]):
            self.logger.info('(%s) || (%s)' % (real, hyp))

        avg_loss = total_loss / n_total_words
        avg_accu = n_total_correct / n_total_words

        self.logger.info('\n')
        self.logger.info('(Validation) ppl: %8.5f, accuracy: %3.3f%%, BLEU %2.2f' % (
            math.exp(min(avg_loss, 100)), 100 * avg_accu, bleu))

    def batch_loss_backward(self, criterion, hyp, gold, step=4):
        # flatten variables
        hyp = hyp.view(-1, hyp.size(2))  # (batch*len, n_vocab)
        gold = gold.contiguous().view(-1)  # (batch*len)

        total_pred = hyp.size(0)
        batch_loss = 0
        # for batch_i, dec in enumerate(hyp.chunk(batch_size)):
        for i in range(0, total_pred, step):
            bound = i + step if (i + step) < total_pred else total_pred
            loss = criterion(hyp[i:bound], gold[i:bound])

            loss /= total_pred
            loss.backward(retain_graph=True)
            batch_loss += loss.data[0]

        return batch_loss

    def train(self, model, train_iter, criterion, optimizer, opt):
        '''Epoch operation in training phase'''
        model.train()

        total_loss = 0
        n_total_words = 0
        n_total_correct = 0

        for batch in tqdm(train_iter, mininterval=2,
                          desc='(Training)', leave=False):
            # prepare data
            src_seq = batch.src
            tgt_seq = batch.tgt
            gold = tgt_seq[:, 1:]

            # forward
            optimizer.zero_grad()
            logit = model(src_seq, tgt_seq)

            # backward
            # loss = self.batch_loss_backward(model, criterion, logit, gold)
            # loss.backward()
            loss, n_correct = self.get_performance(criterion, logit, gold)
            loss.backward()
            # for loss in model.batch_loss(criterion, dec_output, gold):
            #     loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.update_learning_rate()

            # note keeping
            # dec_output = autograd.Variable(dec_output.data, volatile=True)
            # logit = model.tgt_word_proj(dec_output)
            # logit = logit.view(-1, logit.size(2))
            # n_correct = self.get_performance(crit=criterion, pred=logit, gold=gold)

            n_words = gold.data.ne(Constants.PAD).sum()
            n_total_words += n_words
            n_total_correct += n_correct
            total_loss += loss.data[0]

        train_loss = total_loss / n_total_words
        train_accu = n_total_correct / n_total_words
        print('(Training)   ppl: %8.5f, accuracy: %3.3f %%' % (
            math.exp(min(train_loss, 100)), 100 * train_accu))

    def _train(self, model, training_data, validation_data,
               crit, optimizer, opt, TGT_FIELD):
        ''' Start training '''

        log_train_file = None
        log_valid_file = None

        if opt.log:
            log_train_file = opt.log + '.train.log'
            log_valid_file = opt.log + '.valid.log'

            print('[Info] Training performance will be written to file: {} and {}'.format(
                log_train_file, log_valid_file))

            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch,loss,ppl,accuracy\n')
                log_vf.write('epoch,loss,ppl,accuracy\n')

        valid_accus = []
        for epoch_i in range(opt.epoch):
            print('[ Epoch', epoch_i, ']')

            train_iter, val_iter = torchtext.data.BucketIterator.splits(
                (training_data, validation_data),
                batch_sizes=(opt.batch_size, opt.batch_size), device=opt.device,
                sort_key=lambda x: len(x.src), repeat=False)

            start = time.time()
            train_loss, train_accu = self.train_epoch(model, train_iter, crit, optimizer)
            print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
                  'elapse: {elapse:3.3f} min'.format(
                ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu,
                elapse=(time.time() - start) / 60))

            start = time.time()
            valid_loss, valid_accu, bleu = self.evaluate(model, val_iter, crit, TGT_FIELD)
            print('  - (Validation) ppl: %8.5f, accuracy: %3.3f%%, BLEU %2.2f, elapse: %3.3f min' % (
                math.exp(min(valid_loss, 100)), 100 * valid_accu, bleu, (time.time() - start) / 60))

            valid_accus += [valid_accu]

            # save model
            Checkpoint(model=model, optimizer=None, epoch=epoch_i, step=0,
                       input_vocab=None, output_vocab=None).save('./experiment/transformer')

            if log_train_file and log_valid_file:
                with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                    log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                        epoch=epoch_i, loss=train_loss,
                        ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu))
                    log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                        epoch=epoch_i, loss=valid_loss,
                        ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu))
