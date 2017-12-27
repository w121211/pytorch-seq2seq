import random

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable


def pad_seq(seq, max_len, pad_value):
    if seq.dim() != 2:
        raise ValueError('pad sequence只接受 size = (batch, length)')
    batch_size = seq.size(0)
    padded = seq.new(batch_size, max_len).fill_(pad_value)
    for i in range(batch_size):
        padded[i, :seq.size(1)] = seq[i, :]
    return padded


def pad_bos(seq, bos_value):
    batch_size = seq.size(0)
    padded = seq.new(batch_size, seq.size(1) + 1).fill_(bos_value)
    for i in range(batch_size):
        padded[i, 1:] = seq[i, :]
    return padded


def stack(tensor, n_stack, dim=0):
    return torch.cat([tensor for _ in range(n_stack)], dim=dim)


def inflate(tensor, times, dim):
    """
    Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)

    Args:
        tensor: A :class:`Tensor` to inflate
        times: number of repetitions
        dimension: axis for inflation (default=0)

    Returns:
        A :class:`Tensor`

    Examples::
        >> a = torch.LongTensor([[1, 2], [3, 4]])
        >> a
        1   2
        3   4
        [torch.LongTensor of size 2x2]
        >> decoder = TopKDecoder(nn.RNN(10, 20, 2), 3)
        >> b = decoder._inflate(a, 1, dimension=1)
        >> b
        1   1   2   2
        3   3   4   4
        [torch.LongTensor of size 2x4]
        >> c = decoder._inflate(a, 1, dimension=0)
        >> c
        1   2
        1   2
        3   4
        3   4
        [torch.LongTensor of size 4x2]

    """
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)


class DiscriminatorDataPool(object):
    '''
    Examples:
        t1 = torch.LongTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        t2 = torch.LongTensor([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])

        pool = DiscriminatorDataPool(30, 3, 0)
        pool.append_real(t1, to_partial=True)
        pool.append_fake(t2, to_partial=True)
        [(x.seq, x.label) for x in pool.batch_gen(batch_size=4)]
    '''

    class SimpleBatch(object):
        def __init__(self, seq, label):
            self.seq = seq
            self.label = label

    def __init__(self, max_len, min_len, PAD):
        self.fakes = []
        self.reals = []
        self.max_len = max_len
        self.min_len = min_len
        self.PAD = PAD

    def reset(self):
        self.fakes = []
        self.reals = []

    def _append(self, pool, seq, label, to_partial):
        seq = seq[:, :self.max_len]
        if to_partial:
            for i in range(2, seq.size(1) + 1):
                partial = seq[:, :i]
                pool += [(seq, seq.size(1), label) for seq in partial.chunk(partial.size(0))]
        else:
            pool += [(seq, seq.size(1), label) for seq in seq.chunk(seq.size(0))]
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

    def fill(self, data_iter):
        for batch in data_iter:
            self.append_fake(batch.src[0])  # 假設src為假，prob=0
            self.append_real(batch.tgt[0])  # 假設tgt為真，prob=1

    def batch_gen(self, batch_size=16):
        examples = self.reals + self.fakes

        # TODO: 有需要shuffle嗎？
        random.shuffle(examples)

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
            bound = max(lengths) if max(lengths) > self.min_len else self.min_len
            batch_seq = batch_seq[:, :bound]  # 將多餘的部分去掉

            yield self.SimpleBatch(batch_seq, batch_label)


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = len(logits.size())
    gumbel_noise = sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / tau, dims - 1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


# logits = torch.FloatTensor([[10, 2, 0]])
# logits_var = Variable(logits, requires_grad=True)
# y_draw = gumbel_softmax(logits_var, hard=True)
# err = y_draw - Variable(logits.new([[0, 0.5, 1]]))
# loss = (err * err).sum()
# loss.backward()
#
# print(y_draw)
# print(err)
# print(loss)
# print(logits_var.grad)
# print(y_draw.grad)

