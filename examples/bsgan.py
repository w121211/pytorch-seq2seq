import logging
import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
from torchtext import data
from torchtext import datasets

pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if pardir not in sys.path:
    sys.path.insert(0, pardir)

from seq2seq.util.checkpoint import Checkpoint

from ape import Constants, options, helper
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


def load_model(exp_path):
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
    opt.exp_dir = './experiment/transformer-reinforce/use_billion'
    opt.load_vocab_from = './experiment/transformer/lang8-cor2err/vocab.pt'
    opt.build_vocab_from = './data/billion/billion.30m.model.vocab'

    opt.load_D_from = opt.exp_dir

    # dataset params
    opt.max_len = 20

    # G params
    # opt.load_G_a_from = './experiment/transformer/lang8-err2cor/'
    # opt.load_G_b_from = './experiment/transformer/lang8-cor2err/'
    opt.d_word_vec = 300
    opt.d_model = 300
    opt.d_inner_hid = 600
    opt.n_head = 6
    opt.n_layers = 3
    opt.embs_share_weight = True
    opt.beam_size = 1
    opt.max_token_seq_len = opt.max_len + 2  # 包含<BOS>, <EOS>

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

    G = build_G(opt, EN, EN)

    # 預先訓練D
    if opt.load_D_from:
        D = load_model(opt.load_D_from)
    else:
        D = build_D(opt, EN)

    optim_D = torch.optim.Adam(D.parameters(), lr=1e-4)
    optim_G = optim.Adam(G.get_trainable_parameters(),
                         betas=(0.9, 0.98), eps=1e-09)

    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit_G = get_criterion(len(EN.vocab))
    crit_D = nn.BCELoss()

    if opt.cuda:
        G.cuda()
        D.cuda()
        crit_G.cuda()
        crit_D.cuda()

    # ---------- train ----------

    trainer_G = trainers.TransformerTrainer()
    trainer_D = trainers.DiscriminatorTrainer()

    if not opt.load_D_from:
        for epoch in range(1):
            logging.info('[Pretrain Epoch %d]' % epoch)
            pool = helper.DiscriminatorDataPool(opt.max_len, D.min_len, Constants.PAD)

            # 將資料塞進pool中
            train_iter = data.BucketIterator(
                dataset=train, batch_size=opt.batch_size, device=opt.device,
                sort_key=lambda x: len(x.src), repeat=False)

            for step, batch in enumerate(tqdm(train_iter)):
                real_a = batch.src
                real_b = batch.tgt
                pool.append_fake(real_a)  # 假設a為假，prob=0
                pool.append_real(real_b)  # 假設b為真，prob=1

            trainer_D.train(D, train_iter=pool.batch_gen(),
                            crit=crit_D, optimizer=optim_D)
            pool.reset()
        Checkpoint(model=D, optimizer=optim_D, epoch=0, step=0,
                   input_vocab=EN.vocab, output_vocab=EN.vocab).save(opt.exp_dir)

    def eval_G(model):
        _, val_iter = data.BucketIterator.splits(
            (train_lang8, val_lang8), batch_sizes=(opt.batch_size, 128), device=opt.device,
            sort_key=lambda x: len(x.src), repeat=False)
        trainer_G.evaluate(model, val_iter, crit_G, EN)

    # 正式訓練G
    for epoch in range(10):
        logging.info('[Epoch %d]' % epoch)

        train_iter = data.BucketIterator(
            dataset=train, batch_size=opt.batch_size, device=opt.device,
            sort_key=lambda x: len(x.src), repeat=False)

        for step, batch in enumerate(tqdm(train_iter)):
            real_a = batch.src
            gold = real_a[:, 1:]

            optim_G.zero_grad()
            logit = G(real_a, real_a)
            pred = logit.view(-1, logit.size(2))  # (batch*len, n_vocab)
            loss_rest = crit_G(pred, gold.contiguous().view(-1))

            loss_pg = PGloss(G, D, real_a)
            loss = 0.9 * loss_rest + 0.1 * loss_pg

            loss.backward()
            optim_G.step()

            pred = pred.max(1)[1]
            # gold = gold.contiguous().view(-1)
            # n_correct = pred.data.eq(gold.data)
            # n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum()

            if step % 100 == 0:
                print('ppl: %d' % loss)
                print('%s -> %s' % (EN.reverse(gold.data)[0], EN.reverse(pred.data)[0]))


def bsgan_loss(G, D, src_seq):
    fake = G.translate(src_seq)
    D_fake = D(fake)
    loss = 0.5 * torch.mean((log(D_fake) - log(1 - D_fake)) ** 2)
    return loss


def log(x):
    return torch.log(x + 1e-8)


if __name__ == '__main__':
    main()
