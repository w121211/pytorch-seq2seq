from collections import Counter, OrderedDict

import dill

import torch
from torchtext import vocab
from torchtext.data.field import Field


class SentencePieceField(Field):
    # def __init__(self, load_vocab_from=None, **kwargs):
    #     if load_vocab_from is not None:
    #         self.vocab = self.load(load_vocab_from)
    #     super(SentencePieceField, self).__init__(**kwargs)

    def load_vocab(self, path):
        with open(path, 'rb') as f:
            self.vocab = dill.load(f)

    def save_vocab(self, path):
        with open(path, 'wb') as f:
            dill.dump(self.vocab, f)

    def build_vocab_from(self, vocab_path, **kwargs):
        count = dict()
        with open(vocab_path, 'r') as f:
            for line in f:
                token, cnt = line.split('\t')
                count[token] = int(float(cnt) * 10000) + 1000000
        counter = Counter(count)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def reverse(self, batch):
        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        # batch = [filter(filter_special, ex) for ex in batch]
        # if self.use_revtok:
        #     return [revtok.detokenize(ex) for ex in batch]
        return [''.join(ex) for ex in batch]


class BPEmb(vocab.Vectors):
    '''https://github.com/bheinzerling/bpemb'''

    url = {
        'op1000': 'http://cosyne.h-its.org/bpemb/data/en/en.wiki.bpe.op1000.d25.w2v.txt.tar.gz',
        'op10000': 'http://cosyne.h-its.org/bpemb/data/en/en.wiki.bpe.op10000.d300.w2v.txt.tar.gz',
        'op25000': 'http://cosyne.h-its.org/bpemb/data/en/en.wiki.bpe.op25000.d300.w2v.txt.tar.gz',
    }

    def __init__(self, name='op1000', dim=25, **kwargs):
        url = self.url[name]
        name = 'en.wiki.bpe.{}.d{}.w2v.txt'.format(name, str(dim))
        # en.wiki.bpe.op1000.d25.w2v.txt
        super(BPEmb, self).__init__(name, url=url, **kwargs)
