import os
import sys
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
from torchtext import data
from torchtext import datasets

seq2seq_pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if seq2seq_pardir not in sys.path:
    sys.path.insert(0, seq2seq_pardir)

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import NLLLoss, Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

from ape import Constants
from ape.dataset.field import SentencePieceField

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(
        os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # ---------- prepare dataset ----------

    def len_filter(example):
        return len(example.src) <= opt.max_len and len(example.tgt) <= opt.max_len


    EN = SentencePieceField(init_token=Constants.BOS_WORD,
                            eos_token=Constants.EOS_WORD,
                            batch_first=True,
                            include_lengths=True)

    train = datasets.TranslationDataset(
        path='./data/dualgan/train',
        exts=('.billion.sp', '.use.sp'), fields=[('src', EN), ('tgt', EN)],
        filter_pred=len_filter)
    val = datasets.TranslationDataset(
        path='./data/dualgan/val',
        exts=('.billion.sp', '.use.sp'), fields=[('src', EN), ('tgt', EN)],
        filter_pred=len_filter)
    train_lang8, val_lang8 = Lang8.splits(
        exts=('.err.sp', '.cor.sp'), fields=[('src', EN), ('tgt', EN)],
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

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    loss = NLLLoss(size_average=False)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 512
        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size, n_layers=1,
                             bidirectional=bidirectional, variable_lengths=True)
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else 1, n_layers=1,
                             dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        scheduler = StepLR(optimizer.optimizer, 1)
        optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=32,
                          checkpoint_every=1000,
                          print_every=10, expt_dir=opt.expt_dir)

    optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
    seq2seq = t.train(seq2seq, train,
                      num_epochs=20, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)

predictor = Predictor(seq2seq, input_vocab, output_vocab)

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
