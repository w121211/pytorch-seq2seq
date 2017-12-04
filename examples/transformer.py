import logging
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

seq2seq_pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if seq2seq_pardir not in sys.path:
    sys.path.insert(0, seq2seq_pardir)

from seq2seq.models.transformer.Models import Transformer
from seq2seq.models.transformer.Optim import ScheduledOptim
from seq2seq.models.transformer import Constants
from seq2seq.trainer import trainers
from seq2seq.dataset.lang8 import Lang8
from seq2seq.util.checkpoint import Checkpoint

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(filename='exp.transformer.log', format=LOG_FORMAT, level=logging.DEBUG)
# logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('-data', required=True)

    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-resume', action='store_true')

    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                            decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # 快速變更設定
    opt.batch_size = 64
    opt.cuda = torch.cuda.is_available()
    opt.epoch = 10

    opt.d_word_vec = 300
    opt.d_model = 300
    opt.d_inner_hid = 600

    opt.n_head = 5
    opt.n_layers = 3

    opt.embs_share_weight = True

    opt.beam_size = 1

    opt.max_len = 50
    opt.max_token_seq_len = opt.max_len + 2  # 包含<BOS>, <EOS>

    opt.device = None if torch.cuda.is_available() else -1

    # ---------- prepare dataset ----------

    def len_filter(example):
        return len(example.src) <= opt.max_len and len(example.tgt) <= opt.max_len

    EN = data.ReversibleField(init_token=Constants.BOS_WORD,
                              eos_token=Constants.EOS_WORD,
                              batch_first=True)
    train, val = Lang8.splits(
        exts=('.err.bpe', '.cor.bpe'), fields=[('src', EN), ('tgt', EN)],
        train='train', validation='val', test=None, filter_pred=len_filter)
    EN.build_vocab(train, vectors=[GloVe(name='840B', dim='300'), CharNGram(), FastText()])
    logging.info('vocab len: %d' % len(EN.vocab))

    # 檢查Constants是否有誤
    assert EN.vocab.stoi[EN.init_token] == Constants.BOS
    assert EN.vocab.stoi[EN.eos_token] == Constants.EOS
    assert EN.vocab.stoi[EN.pad_token] == Constants.PAD
    assert EN.vocab.stoi[EN.unk_token] == Constants.UNK

    # ---------- init model ----------

    logging.info(opt)

    transformer = Transformer(
        len(EN.vocab),
        len(EN.vocab),
        opt.max_token_seq_len,
        proj_share_weight=opt.proj_share_weight,
        embs_share_weight=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        encoder_emb_weight=EN.vocab.vectors,
        decoder_emb_weight=EN.vocab.vectors, )

    # print(transformer)

    optimizer = ScheduledOptim(
        optim.Adam(
            transformer.get_trainable_parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit = get_criterion(len(EN.vocab))

    if torch.cuda.is_available():
        transformer.cuda()
        crit.cuda()

    # ---------- training ----------

    trainer = trainers.TransformerTrainer()

    for epoch in range(opt.epoch):
        logging.info('[Epoch %d]' % epoch)

        train_iter, val_iter = data.BucketIterator.splits(
            (train, val), batch_sizes=(opt.batch_size, opt.batch_size), device=opt.device,
            sort_key=lambda x: len(x.src), repeat=False)

        trainer.train_epoch(transformer, train_iter, crit, optimizer, opt)
        trainer.evaluate(transformer, val_iter, crit, EN)

        Checkpoint(model=transformer, optimizer=optimizer, epoch=epoch, step=0,
                   input_vocab=EN.vocab, output_vocab=EN.vocab).save('./experiment/transformer')


if __name__ == '__main__':
    main()
