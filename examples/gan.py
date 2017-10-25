import os
import sys
import inspect
import argparse
import logging

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchtext

seq2seq_pardir = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if seq2seq_pardir not in sys.path:
    sys.path.insert(0, seq2seq_pardir)

import seq2seq as SEQ2SEQ
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.beam import Beam

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR




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

opt.train_path = './data/train.txt'
# opt.batch_size = 1
# opt.sample_size = 3

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
    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    max_len = 60


    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len


    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter)
    # dev = torchtext.data.TabularDataset(
    #     path=opt.dev_path, format='tsv',
    #     fields=[('src', src), ('tgt', tgt)],
    #     filter_pred=len_filter
    # )
    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    # seq2seq = None
    optimizer = None
    encoder = None
    decoder = None
    if not opt.resume:
        # Initialize model
        hidden_size = 128
        bidirectional = False
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                             bidirectional=bidirectional, variable_lengths=False)
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                             dropout_p=0, use_attention=False, bidirectional=bidirectional,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id)

        # seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            encoder.cuda()
            decoder.cuda()
            # seq2seq.cuda()
        # for param in seq2seq.parameters():
        #     param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2)

    # policy gradient training
    batch_size = 1  # reinforce 僅允許1次1個
    batch_size = 1  # reinforce 僅允許1次1個
    sample_size = 3

    device = None if torch.cuda.is_available() else -1
    batch_iterator = torchtext.data.BucketIterator(
        dataset=train, batch_size=batch_size,
        sort_key=lambda x: -len(x.src),
        device=device, repeat=False)
    batch_generator = iter(batch_iterator)

    for batch in batch_generator:
        # (1) encode
        input_variables, input_lengths = getattr(batch, SEQ2SEQ.src_field_name)
        target_variables = getattr(batch, SEQ2SEQ.tgt_field_name)
        # print(input_variables)
        # print(input_lengths)
        encoder_outputs, encoder_hidden = encoder(input_variables, input_lengths)
        # print(encoder_outputs)

        # (2) decode, sampling, policy gradient training
        for _ in range(sample_size):  # 一個sample視為一個episode
            inputs = Variable(torch.LongTensor([tgt.sos_id])).view(batch_size, -1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            actions = []
            decodes = []

            decoder_input = inputs[:, 0].unsqueeze(1)
            hidden = encoder_hidden
            for i in range(max_len):
                probs, hidden, attn = decoder.forward_step(decoder_input, hidden, None, F.log_softmax)

                # decode
                probs = probs.exp().squeeze(0)
                action = probs.multinomial(1)  # (batch, 1), 從vocab probs中隨機選出一個vocab_id
                symbol = action.data[0, 0]
                actions.append(action)
                decodes.append((symbol, probs.data[0, symbol]))  # (vocab_id, prob_vocab)
                decoder_input = action
                if symbol == tgt.eos_id:
                    break
            # print(actions)

            # 先算整體reward，再計算每個action的reward
            # reward = discriminator(seq)
            reward = 10
            for action in actions:
                # r = action * reward  # log-prob * reward
                # print(action.grad_fn)
                action.reinforce(reward)
            optimizer.zero_grad()
            autograd.backward(actions, [None for _ in actions], retain_graph=True)
            optimizer.step()
            del actions

            # print([p for p in decoder.parameters()])


def main():
    # init dataset

    # init model
    hidden_size = 128
    bidirectional = True
    encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                         bidirectional=bidirectional, variable_lengths=True)
    decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else 1,
                         dropout_p=0, use_attention=False, bidirectional=bidirectional,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)

    generator = Seq2seq(encoder, decoder)
    if torch.cuda.is_available():
        generator.cuda()
    for param in generator.parameters():
        param.data.uniform_(-0.08, 0.08)

    # pre-train generator


    # pre-train discriminator

    # adversarial training


    for epoch in range(opt.adv_train_epochs):
        # generate samples
        samples = generator.sample()

        # rewards

        # train generator with policy gradient


        # train discriminator


def beam():
    beams = [Beam(5, n_best=5, cuda=torch.cuda.is_available(),
                  vocab=output_vocab, global_scorer=None) for _ in range(opt.batch_size)]
    for i in range(max_len):
        inputs = torch.stack(
            [b.get_current_state() for b in beam if not b.done]
        ).t().contiguous().view(1, -1)

        active = []
        for b in range(batch_size):
            if beams[b].done:
                continue

            if not beams[b].advance():
                active += [b]

        if len(active) == 0:
            break

# predictor = Predictor(seq2seq, input_vocab, output_vocab)

# while True:
#     seq_str = raw_input("Type in a source sequence:")
#     seq = seq_str.strip().split()
#     print(predictor.predict(seq))
