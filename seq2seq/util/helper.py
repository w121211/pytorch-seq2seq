import random
import numpy as np

import torch
import torch.autograd as autograd

import seq2seq
from seq2seq.models.transformer import Constants
from .checkpoint import Checkpoint


def get_position(seq_tensor):
    pos = np.array([
        [pos_i + 1 if token_i != Constants.PAD else 0
         for pos_i, token_i in enumerate(seq)]
        for seq in seq_tensor])
    return autograd.Variable(torch.LongTensor(pos))


def pad_sequence(seq, max_len, pad_value=1):
    if seq.dim() != 2:
        raise ValueError('pad sequence只接受 size = (batch, length)')

    batch_size = seq.size(0)
    padded = seq.new(batch_size, max_len).fill_(pad_value)
    for i in range(batch_size):
        padded[i, :seq.size(1)] = seq[i, :]
    return padded


class DDataPool(object):
    class SimpleBatch(object):
        def __init__(self, seq, label):
            self.seq = seq
            self.label = label

    def __init__(self, max_len):
        self.fakes = []
        self.reals = []
        self.max_len = max_len

    def reset(self):
        self.fakes = []
        self.reals = []

    def append_fake(self, fake_tensor):
        self.fakes += [(seq, 0) for seq in fake_tensor.chunk(fake_tensor.size(0))]

    def append_real(self, real_tensor):
        self.reals += [(seq, 1) for seq in real_tensor.chunk(real_tensor.size(0))]

    def batch_gen(self, batch_size=16):
        examples = self.reals + self.fakes

        # drop out
        random.shuffle(examples)
        if batch_size < len(examples):
            examples = examples[0:batch_size]

        for i in range(0, len(examples), batch_size):
            step_size = batch_size if i + batch_size < len(examples) else len(examples) - i

            seq = autograd.Variable(torch.LongTensor(step_size, self.max_len).fill_(seq2seq.pad_id))
            label = autograd.Variable(torch.zeros(step_size).float())
            if torch.cuda.is_available():
                seq = seq.cuda()
                label = label.cuda()

            for j, (_seq, _label) in enumerate(examples[i:i + step_size]):
                seq[j] = _seq
                label[j] = _label
            yield self.SimpleBatch(seq, label)


def batch_gen(fakes, reals, max_len, batch_size=16):
    '''
    非常naive的batch generator，用於將samples與reals的tensors合併、shuffle、然後產生batches。

    Arguments:
        fakes (Tensor): (batch, seq_len)
        reals (Tensor): (batch, seq_len)

    return:
        Variable (batch, max_seq_len) naive的將所有不足長度的seq全部pad成max_len
    '''
    reals = [(src, src.size(1), 1) for src in reals.chunk(reals.size(0))]  # tuple: (Tensor, seq_len, tgt_label)
    samples = [(src, src.size(1), 0) for src in fakes]

    examples = reals + samples
    random.shuffle(examples)

    for i in range(0, len(examples), batch_size):
        if torch.cuda.is_available():
            src = autograd.Variable(torch.LongTensor(batch_size, max_len).fill_(seq2seq.pad_id)).cuda()
            tgt = autograd.Variable(torch.zeros(batch_size, 1).float()).cuda()
        else:
            src = autograd.Variable(torch.LongTensor(batch_size, max_len).fill_(seq2seq.pad_id))
            tgt = autograd.Variable(torch.zeros(batch_size, 1).float())

        for j, (seq, seq_len, label) in enumerate(examples[i:i + batch_size]):
            src[j, :seq_len] = seq
            tgt[j, 0] = label
        yield SimpleBatch(batch_size, seq, label)


class MyCheckpoint(Checkpoint):
    def save(self):
        pass
