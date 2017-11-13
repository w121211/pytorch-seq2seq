import random

import torch
import torch.autograd as autograd


class SimpleBatch(object):
    def __init__(self, batch_size, src, tgt):
        self.batch_size = batch_size
        self.src = src
        self.tgt = tgt


def batch_gen(samples, reals, pad_id, max_len, batch_size=16):
    '''
    非常naive的batch generator，用於將samples與reals的tensors合併、shuffle、然後產生batches。

    argument:
        samples list of Tensors
        reals  a Tensor (batch, seq_len)

    return:
        Variable (batch, max_seq_len) naive的將所有不足長度的seq全部pad成max_len
    '''
    reals = [(src, src.size(1), 1) for src in reals.chunk(reals.size(0))]  # tuple: (Tensor, seq_len, tgt_label)
    samples = [(src, src.size(1), 0) for src in samples]

    examples = reals + samples
    random.shuffle(examples)

    for i in range(0, len(examples), batch_size):
        if torch.cuda.is_available():
            src = autograd.Variable(torch.LongTensor(batch_size, max_len).fill_(pad_id)).cuda()
            tgt = autograd.Variable(torch.zeros(batch_size, 1).float()).cuda()
        else:
            src = autograd.Variable(torch.LongTensor(batch_size, max_len).fill_(pad_id))
            tgt = autograd.Variable(torch.zeros(batch_size, 1).float())

        for j, (seq, seq_len, label) in enumerate(examples[i:i + batch_size]):
            src[j, :seq_len] = seq
            tgt[j, 0] = label
        yield SimpleBatch(batch_size, src, tgt)
