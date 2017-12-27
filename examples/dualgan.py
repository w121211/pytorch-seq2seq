import logging
import os
import sys
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data
from torchtext import datasets

pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if pardir not in sys.path:
    sys.path.insert(0, pardir)

from seq2seq.util.checkpoint import Checkpoint

from ape import Constants, options
from ape.dataset.lang8 import Lang8
from ape.dataset.field import SentencePieceField
from ape.model.discriminator import BinaryClassifierCNN
from ape.model.transformer.Models import Transformer
from ape import trainers

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'


def build_G(opt, SRC_FIELD, TGT_FIELD):
    return Transformer(
        len(SRC_FIELD.vocab),
        len(TGT_FIELD.vocab),
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
        encoder_emb_weight=SRC_FIELD.vocab.vectors,
        decoder_emb_weight=TGT_FIELD.vocab.vectors, )


def build_D(opt, TEXT_FIELD):
    return BinaryClassifierCNN(len(TEXT_FIELD.vocab),
                               embed_dim=opt.embed_dim,
                               num_kernel=opt.num_kernel,
                               kernel_sizes=opt.kernel_sizes,
                               dropout_p=opt.dropout_p)


def load_G(exp_path):
    cp = Checkpoint.load(Checkpoint.get_latest_checkpoint(exp_path))
    model = cp.model
    return model


def main():
    parser = argparse.ArgumentParser()
    opt = options.train_options(parser)
    opt = parser.parse_args()

    opt.cuda = torch.cuda.is_available()
    opt.device = None if opt.cuda else -1

    # 快速變更設定
    opt.exp_dir = './experiment/transformer-dualgan/use_billion'
    opt.load_vocab_from = './experiment/transformer/lang8-cor2err/vocab.pt'
    opt.build_vocab_from = './data/billion/billion.30m.model.vocab'

    # dataset params
    opt.max_len = 20

    # G params
    opt.load_G_a_from = './experiment/transformer/lang8-err2cor/'
    opt.load_G_b_from = './experiment/transformer/lang8-cor2err/'
    opt.d_model = 300  # 暫時需要

    # D params
    opt.embed_dim = opt.d_model
    opt.num_kernel = 100
    opt.kernel_sizes = [2, 3, 4, 5, 6, 7]
    opt.dropout_p = 0.25

    # train params
    opt.batch_size = 1
    opt.n_epoch = 5

    if not os.path.exists(opt.exp_dir):
        os.makedirs(opt.exp_dir)
    logging.basicConfig(filename=opt.exp_dir + '/.log',
                        format=LOG_FORMAT, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('Use CUDA? ' + str(opt.cuda))
    logging.info(opt)

    # ---------- prepare dataset ----------

    def len_filter(example):
        return len(example.src) <= opt.max_len and len(example.tgt) <= opt.max_len

    EN = SentencePieceField(init_token=Constants.BOS_WORD,
                            eos_token=Constants.EOS_WORD,
                            batch_first=True)

    train = datasets.TranslationDataset(
        path='./data/dualgan/train',
        exts=('.billion.sp', '.use.sp'), fields=[('src', EN), ('tgt', EN)],
        filter_pred=len_filter)
    # 用於 evaluate G_a
    train_lang8, val_lang8 = Lang8.splits(
        exts=('.err.tiny.sp', '.cor.tiny.sp'), fields=[('src', EN), ('tgt', EN)],
        train='test', validation='test', test=None, filter_pred=len_filter)

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

    G_a = load_G(opt.load_G_a_from)
    G_b = load_G(opt.load_G_b_from)
    D_a = build_D(opt, EN)
    D_b = build_D(opt, EN)

    optim_G_a = optim.Adam(G_a.get_trainable_parameters(),
                           betas=(0.9, 0.98), eps=1e-09)
    optim_G_b = optim.Adam(G_a.get_trainable_parameters(),
                           betas=(0.9, 0.98), eps=1e-09)
    optim_D_a = torch.optim.Adam(D_a.parameters(), lr=1e-4)
    optim_D_b = torch.optim.Adam(D_b.parameters(), lr=1e-4)

    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit_G = get_criterion(len(EN.vocab))
    crit_D = nn.BCELoss()

    if opt.cuda:
        G_a.cuda()
        G_b.cuda()
        D_a.cuda()
        D_b.cuda()
        crit_G.cuda()
        crit_D.cuda()

    # ---------- train ----------

    trainer_G = trainers.TransformerTrainer()
    trainer = trainers.DualGanPGTrainer(
        opt,
        trainer_G=trainer_G,
        trainer_D=trainers.DiscriminatorTrainer())

    def eval_G(model):
        _, val_iter = data.BucketIterator.splits(
            (train_lang8, val_lang8), batch_sizes=(opt.batch_size, 128), device=opt.device,
            sort_key=lambda x: len(x.src), repeat=False)
        trainer_G.evaluate(model, val_iter, crit_G, EN)

    for epoch in range(10):
        logging.info('[Epoch %d]' % epoch)

        train_iter = data.BucketIterator(
            dataset=train, batch_size=opt.batch_size, device=opt.device,
            sort_key=lambda x: len(x.src), repeat=False)
        # batch = next(iter(train_iter))
        # src_seq = batch.src
        # tgt_seq = batch.tgt

        trainer.train(
            0,
            train_iter,
            G_a=G_a,
            G_b=G_b,
            D_a=D_a,
            D_b=D_b,
            optim_G_a=optim_G_a,
            optim_G_b=optim_G_b,
            optim_D_a=optim_D_a,
            optim_D_b=optim_D_b,
            crit_G=crit_G,
            crit_D=crit_D,
            eval_G=eval_G,
            A_FIELD=EN,
            B_FIELD=EN)

        # for epoch in range(opt.n_epoch):
        #     logging.info('[Epoch %d]' % epoch)
        #
        #     train_iter, val_iter = data.BucketIterator.splits(
        #         (train, val), batch_sizes=(opt.batch_size, opt.batch_size), device=opt.device,
        #         sort_key=lambda x: len(x.src), repeat=False)
        #
        #     trainer.train(transformer, train_iter, crit, optimizer, opt)
        #     # trainer.evaluate(transformer, val_iter, crit, EN)
        #
        #     Checkpoint(model=transformer, optimizer=optimizer, epoch=epoch, step=0,
        #                input_vocab=EN.vocab, output_vocab=EN.vocab).save('./experiment/transformer')


if __name__ == '__main__':
    main()
