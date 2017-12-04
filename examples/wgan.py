import logging
import os
import sys
import re
import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

seq2seq_pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if seq2seq_pardir not in sys.path:
    sys.path.insert(0, seq2seq_pardir)

from seq2seq.models.gan import TestDiscriminator
from seq2seq.models.transformer.Models import Transformer
from seq2seq.models.transformer.Optim import ScheduledOptim
from seq2seq.models.transformer import Constants, Translator
from seq2seq.trainer import trainers
from seq2seq.trainer.gan import WganTrainer
from seq2seq.dataset.lang8 import Lang8

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
# logging.basicConfig(filename='exp.log', format=LOG_FORMAT, level=logging.DEBUG)
logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
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

    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                            decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # 快速變更設定
    # opt.n_layers = 1
    # opt.batch_size = 4
    opt.cuda = torch.cuda.is_available()
    opt.epoch = 2000
    opt.save_model = 'trained'
    opt.model = 'trained.chkpt'

    opt.d_word_vec = 300
    opt.d_model = 300
    opt.d_inner_hid = 600

    opt.embs_share_weight = True

    opt.beam_size = 1

    opt.max_len = 50
    opt.max_token_seq_len = opt.max_len + 2  # 包含<BOS>, <EOS>

    opt.device = None if torch.cuda.is_available() else -1

    # =========== prepare dataset ===========
    def len_filter(example):
        return len(example.src) <= opt.max_len and len(example.tgt) <= opt.max_len

    EN = data.ReversibleField(init_token=Constants.BOS_WORD,
                              eos_token=Constants.EOS_WORD,
                              batch_first=True)
    train, val = Lang8.splits(
        exts=('.err.bpe', '.cor.bpe'), fields=[('src', EN), ('tgt', EN)],
        train='test', validation='test', test=None, filter_pred=len_filter)
    # adv_train, adv_dev, adv_test = Lang8.splits(
    #     exts=('.adv.cor', '.adv.err'), fields=[('src', src), ('tgt', tgt)],
    #     train='test', validation='test', test='test')
    # BD.build_vocab(train, vectors=[GloVe(name='840B', dim='300'), CharNGram(), FastText()])
    # GD.build_vocab(train, vectors=[GloVe(name='840B', dim='300'), CharNGram(), FastText()])
    EN.build_vocab(train, vectors=FastText())
    print('vocab len: %d' % len(EN.vocab))

    # 檢查Constants是否有誤
    assert EN.vocab.stoi[EN.init_token] == Constants.BOS
    assert EN.vocab.stoi[EN.eos_token] == Constants.EOS
    assert EN.vocab.stoi[EN.pad_token] == Constants.PAD
    assert EN.vocab.stoi[EN.unk_token] == Constants.UNK

    # ---------- init model ----------
    # if opt.embs_share_weight and train.src_word2idx != train.tgt_word2idx:
    #     print('[Warning] The src/tgt word2idx table are different but asked to share word embedding.')

    print(opt)

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

    discriminator = TestDiscriminator(len(EN.vocab),
                                      d_model=300,
                                      max_len=opt.max_token_seq_len, )

    print(transformer)
    print(discriminator)

    optimizer = ScheduledOptim(
        optim.Adam(
            transformer.get_trainable_parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    optimizer_G = optim.Adam(transformer.get_trainable_parameters(),
                             lr=1e-4, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=1e-4, betas=(0.5, 0.9))

    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit = get_criterion(len(EN.vocab))

    if opt.cuda:
        transformer.cuda()
        discriminator.cuda()
        crit.cuda()

    # =========== training ===========
    supervised_trainer = trainers.TransformerTrainer()
    # trainer.train(transformer, train, val, crit, optimizer, opt, GD)

    # train_iter, val_iter = data.BucketIterator.splits(
    #     (train, val), batch_sizes=(4, 256), device=opt.device,
    #     sort_key=lambda x: len(x.src))
    # batch = next(iter(train_iter))
    # src_seq = batch.src
    # tgt_seq = batch.tgt
    # src_pos = transformer.get_position(src_seq.data)
    # tgt_pos = transformer.get_position(tgt_seq.data)
    #
    # # print(tgt_seq)
    # # print(src_pos)
    # # print(tgt_pos)
    #
    # transformer(src_seq, src_pos, tgt_seq, tgt_pos)
    # output = transformer(src_seq, src_pos)
    # print(output)
    #
    # print(discriminator(output))

    # =========== WGAN training ===========
    wgan_trainer = WganTrainer(opt)
    train_iter, val_iter = data.BucketIterator.splits(
        (train, val), batch_sizes=(16, 64), device=opt.device,
        sort_key=lambda x: len(x.src), repeat=False)

    for epoch in range(opt.epoch):
        print('[Epoch %d]' % epoch)
        wgan_trainer.train_epoch(epoch, D=discriminator, G=transformer,
                                 optimizer_D=optimizer_D, optimizer_G=optimizer_G,
                                 train_iter=train_iter, n_tgt_vocab=len(EN.vocab))
        valid_loss, valid_accu, bleu = supervised_trainer.evaluate(transformer, val_iter, crit, EN)
        print('(Validation) ppl: %8.5f, accuracy: %3.3f%%, BLEU %2.2f' % (
            math.exp(min(valid_loss, 100)), 100 * valid_accu, bleu))


if __name__ == '__main__':
    main()
