import logging
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if pardir not in sys.path:
    sys.path.insert(0, pardir)

from seq2seq.util.checkpoint import Checkpoint

from ape import Constants
from ape.model.transformer.Models import Transformer
from ape.model.transformer.Optim import ScheduledOptim
from ape.trainer.seq2seq import TransformerTrainer
from ape.dataset.lang8 import Lang8
from ape.dataset.field import SentencePieceField

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

    parser.add_argument('-exp_dir', type=str, default='./experiment')
    parser.add_argument('-load_from', type=str, default=None)

    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                            decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = torch.cuda.is_available()
    opt.d_word_vec = opt.d_model

    logging.info('Use CUDA? ' + str(opt.cuda))

    # 快速變更設定
    opt.exp_dir = './experiment/transformer/lang8-cor2err'
    opt.load_vocab_from = './experiment/transformer/lang8-cor2err/vocab.pt'
    opt.build_vocab_from = './data/billion/billion.30m.model.vocab'

    opt.batch_size = 64
    opt.cuda = torch.cuda.is_available()
    opt.epoch = 10

    opt.d_word_vec = 300
    opt.d_model = 300
    opt.d_inner_hid = 600

    opt.n_head = 6
    opt.n_layers = 3

    opt.embs_share_weight = True

    opt.beam_size = 1

    opt.max_len = 50
    opt.max_token_seq_len = opt.max_len + 2  # 包含<BOS>, <EOS>

    opt.device = None if torch.cuda.is_available() else -1

    logging.info(opt)

    # ---------- prepare dataset ----------

    def len_filter(example):
        return len(example.src) <= opt.max_len and len(example.tgt) <= opt.max_len

    EN = SentencePieceField(init_token=Constants.BOS_WORD,
                            eos_token=Constants.EOS_WORD,
                            batch_first=True)
    train, val = Lang8.splits(
        exts=('.cor.sp', '.err.sp'), fields=[('src', EN), ('tgt', EN)],
        train='train', validation='val', test=None, filter_pred=len_filter)

    # 讀取 vocabulary（確保一致）
    try:
        logging.info('Load voab from %s' % opt.load_vocab_from)
        EN.load_vocab(opt.load_vocab_from)
    except FileNotFoundError:
        EN.build_vocab_from(opt.build_vocab_from)
        EN.save_vocab(opt.load_vocab_from)

    logging.info('Vocab len: %d' % len(EN.vocab))

    # 檢查Constants是否有誤
    assert EN.vocab.stoi[Constants.BOS_WORD] == Constants.BOS
    assert EN.vocab.stoi[Constants.EOS_WORD] == Constants.EOS
    assert EN.vocab.stoi[Constants.PAD_WORD] == Constants.PAD
    assert EN.vocab.stoi[Constants.UNK_WORD] == Constants.UNK

    # ---------- init model ----------

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

    if opt.cuda:
        transformer.cuda()
        crit.cuda()

    # ---------- training ----------

    trainer = TransformerTrainer()

    for epoch in range(opt.epoch):
        logging.info('[Epoch %d]' % epoch)

        train_iter, val_iter = data.BucketIterator.splits(
            (train, val), batch_sizes=(opt.batch_size, opt.batch_size), device=opt.device,
            sort_key=lambda x: len(x.src), repeat=False)

        trainer.train(transformer, train_iter, crit, optimizer, opt)
        trainer.evaluate(transformer, val_iter, crit, EN)

        Checkpoint(model=transformer, optimizer=optimizer, epoch=epoch, step=0,
                   input_vocab=None, output_vocab=None).save(opt.exp_dir)


if __name__ == '__main__':
    main()
