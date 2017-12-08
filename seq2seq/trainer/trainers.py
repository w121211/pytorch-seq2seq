import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from tqdm import tqdm

from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.models.transformer import Constants
from seq2seq.util.checkpoint import Checkpoint


class BaseTrainer:
    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def train(self, *args, **kwargs):
        raise NotImplementedError()


class BinaryClassifierTrainer(BaseTrainer):
    def __init__(self, loss=nn.BCELoss(), threshold=0.5, *args, **kwargs):
        super(BinaryClassifierTrainer, self).__init__(*args, **kwargs)
        self.loss = loss
        self.threshold = threshold

    def train_epoch(self, model, train_iter, optimizer,
                    dev_iter=None, resume=False, step=0):
        model.train()
        for batch in train_iter:
            feature, target = batch.src, batch.tgt
            optimizer.zero_grad()
            prob = model(feature)
            loss = self.loss(prob, target)
            loss.backward()
            optimizer.step()

            if step % self.print_every == 0:
                corrects = (prob.gt(self.threshold).float().view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / batch.batch_size
                self.logger.info('Step[%d] - loss: %.6f acc: %.4f%%(%d/%d)' % (
                    step, loss.data[0], accuracy, corrects, batch.batch_size))
            step += 1
            # if steps % self.checkpoint_every == 0:
            #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            #     save_prefix = os.path.join(args.save_dir, 'snapshot')
            #     save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            #     torch.save(model, save_path)
        return step

    def evaluate(self, model, data_iter):
        model.eval()
        corrects, avg_loss = 0, 0
        size = 0
        for batch in data_iter:
            feature, target = batch.src, batch.tgt
            prob = model(feature)
            loss = self.loss(prob, target)
            avg_loss += loss.data[0]
            corrects += (prob.gt(self.threshold).float().view(target.size()).data == target.data).sum()
            size += batch.batch_size
        avg_loss = avg_loss / size
        accuracy = 100.0 * corrects / size
        self.logger.info(
            'Evaluation - loss: %.6f  acc: %.4f%%(%d/%d)' % (avg_loss, accuracy, corrects, size))
        return avg_loss, accuracy

    def predict(self, text, model, text_field, label_feild):
        assert isinstance(text, str)
        model.eval()
        # text = text_field.tokenize(text)
        text = text_field.preprocess(text)
        text = [[text_field.vocab.stoi[x] for x in text]]
        x = text_field.tensor_type(text)
        x = autograd.Variable(x, volatile=True)
        print(x)
        output = model(x)
        _, predicted = torch.max(output, 1)
        return label_feild.vocab.itos[predicted.data[0][0] + 1]


class ClassifierTrainer(BaseTrainer):
    def __init__(self, loss=nn.BCELoss(), *args, **kwargs):
        super(ClassifierTrainer, self).__init__(*args, **kwargs)
        self.loss = loss

    def train(self, model, train_iter, optimizer,
              dev_iter=None, num_epochs=5, resume=False):
        model.train()
        step = 0
        for epoch in range(1, num_epochs + 1):
            for batch in train_iter:
                feature, target = batch.src, batch.tgt
                optimizer.zero_grad()
                prob = model(feature)
                print(prob)
                # print(target)
                loss = self.loss(prob, target)
                loss.backward()
                optimizer.step()

                if step % self.print_every == 0:
                    self.logger.info('Batch[%d] - loss: %.6f' % (step, loss.data[0]))

                    # corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                    # accuracy = 100.0 * corrects / batch.batch_size
                    # self.logger.info('Batch[%d] - loss: %.6f acc: %.4f(%d/%d)' % (
                    #     step, output.data[0], accuracy, corrects, batch.batch_size))
                step += 1
                # if steps % args.test_interval == 0:
                #     self.eval(dev_iter, model, args)
                # if steps % self.checkpoint_every == 0:
                #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                #     save_prefix = os.path.join(args.save_dir, 'snapshot')
                #     save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                #     torch.save(model, save_path)

    def evaluate(self, model, data_iter):
        model.eval()
        corrects, avg_loss = 0, 0
        for batch in data_iter:
            feature, target = batch.text, batch.label
            # target.data.sub_(1)  # class label 需為[0 ~ class_num-1]

            logit = model(feature)
            loss = F.cross_entropy(logit, target, size_average=False)

            avg_loss += loss.data[0]
            corrects += (torch.max(logit, 1)
                         [1].view(target.size()).data == target.data).sum()

        size = len(data_iter.dataset)
        avg_loss = avg_loss / size
        accuracy = 100.0 * corrects / size
        # model.train()
        self.logger.info('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                                      accuracy,
                                                                                      corrects,
                                                                                      size))

    def predict(self, text, model, text_field, label_feild):
        assert isinstance(text, str)
        model.eval()
        # text = text_field.tokenize(text)
        text = text_field.preprocess(text)
        text = [[text_field.vocab.stoi[x] for x in text]]
        x = text_field.tensor_type(text)
        x = autograd.Variable(x, volatile=True)
        print(x)
        output = model(x)
        _, predicted = torch.max(output, 1)
        return label_feild.vocab.itos[predicted.data[0][0] + 1]


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
        ''' Epoch operation in evaluation phase '''
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
            src_pos = model.get_position(src_seq.data)
            tgt_seq = batch.tgt
            tgt_pos = model.get_position(tgt_seq.data)
            gold = tgt_seq[:, 1:]

            # forward
            src_seq.volatile = True
            logit = model(src_seq, src_pos, tgt_seq, tgt_pos)
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

            hyps += TGT_FIELD.reverse(pred.data)
            reals += TGT_FIELD.reverse(tgt_seq.data)

        bleu = bleu.moses_multi_bleu(hypotheses=np.array(hyps),
                                     references=np.array(reals),
                                     lowercase=True)

        for real, hyp in zip(reals[0:10], hyps[0:10]):
            self.logger.info('(%s) || (%s)' % (real, hyp))

        valid_loss = total_loss / n_total_words
        valid_accu = n_total_correct / n_total_words

        self.logger.info('(Validation) ppl: %8.5f, accuracy: %3.3f%%, BLEU %2.2f' % (
            math.exp(min(valid_loss, 100)), 100 * valid_accu, bleu))

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

    def train_epoch(self, model, train_iter, criterion, optimizer, opt):
        ''' Epoch operation in training phase'''
        model.train()

        total_loss = 0
        n_total_words = 0
        n_total_correct = 0

        for batch in tqdm(train_iter, mininterval=2,
                          desc='(Training)   ', leave=False):
            # prepare data
            src_seq = batch.src
            src_pos = model.get_position(src_seq.data)
            tgt_seq = batch.tgt
            tgt_pos = model.get_position(tgt_seq.data)

            gold = tgt_seq[:, 1:]

            # forward
            optimizer.zero_grad()
            logit = model(src_seq, src_pos, tgt_seq, tgt_pos)

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

    def train(self, model, training_data, validation_data,
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
