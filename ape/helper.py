import random

import torch
import torch.autograd as autograd


def pad_seq(seq, max_len, pad_value=1):
    if seq.dim() != 2:
        raise ValueError('pad sequence只接受 size = (batch, length)')

    batch_size = seq.size(0)
    padded = seq.new(batch_size, max_len).fill_(pad_value)
    for i in range(batch_size):
        padded[i, :seq.size(1)] = seq[i, :]
    return padded


def stack_seq(seq, n_stack, dim=0):
    return torch.cat([seq for _ in range(n_stack)], dim=dim)


class DiscriminatorDataPool(object):
    class SimpleBatch(object):
        def __init__(self, seq, label):
            self.seq = seq
            self.label = label

    def __init__(self, max_len, PAD):
        self.fakes = []
        self.reals = []
        self.max_len = max_len
        self.PAD = PAD

    def reset(self):
        self.fakes = []
        self.reals = []

    @staticmethod
    def _append(pool, seq, label, to_partial):
        if to_partial:
            for i in range(2, seq.size(1)):
                partial = seq[:, :i]
                pool += [(seq, seq.size(1), label) for seq in partial.chunk(partial.size(0))]
        else:
            pool += [(seq, label) for seq in seq.chunk(seq.size(0))]
            # return pool

    def append_fake(self, fake_seq, to_partial=False):
        '''
        Args:
            fake_seq: (batch, len)
            to_partial: 將seq依字節生成partial seq，並存至pool
        '''
        self._append(self.fakes, fake_seq, label=0, to_partial=to_partial)

    def append_real(self, real_seq, to_partial=False):
        '''
        Args:
            real_seq: (batch, len)
            to_partial: 將seq依字節生成partial seq，並存至pool
        '''
        self._append(self.reals, real_seq, label=1, to_partial=to_partial)

    def batch_gen(self, batch_size=16):
        examples = self.reals + self.fakes

        # TODO: 有需要shuffle嗎？
        # random.shuffle(examples)
        if batch_size < len(examples):
            examples = examples[0:batch_size]

        for i in range(0, len(examples), batch_size):
            step_size = batch_size if i + batch_size < len(examples) else len(examples) - i

            batch_seq = autograd.Variable(torch.LongTensor(step_size, self.max_len).fill_(self.PAD))
            batch_label = autograd.Variable(torch.zeros(step_size).float())
            if torch.cuda.is_available():
                batch_seq = batch_seq.cuda()
                batch_label = batch_label.cuda()

            lengths = []
            for j, (seq, seq_len, label) in enumerate(examples[i:i + step_size]):
                batch_seq[j, :seq.size(1)] = seq[0, :]
                batch_label[j] = label
                lengths.append(seq_len)
            batch_seq = batch_seq[:, :max(lengths)]  # 將多餘的部分去掉

            yield self.SimpleBatch(batch_seq, batch_label)
