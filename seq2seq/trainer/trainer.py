import os
import sys
import random
import logging

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

seq2seq_pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if seq2seq_pardir not in sys.path:
    sys.path.insert(0, seq2seq_pardir)

from seq2seq.loss import NLLLoss
from seq2seq.evaluator import Evaluator


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
                    dev_iter=None, resume=False):
        model.train()
        step = 0
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
