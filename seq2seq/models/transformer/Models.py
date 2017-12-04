''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

from seq2seq.util import helper
from . import Constants
from .Modules import BottleLinear as Linear
from .Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)  # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # bxsqxsk
    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    if seq.is_cuda:
        subsequent_mask = torch.from_numpy(subsequent_mask).cuda()
    else:
        subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,
                 emb_weight=None):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        if emb_weight is not None:
            self.src_word_emb.weight.data.copy_(emb_weight)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        # Word embedding look up
        enc_input = self.src_word_emb(src_seq)  # (batch, len, embedding_dim)

        # Position Encoding addition
        enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)  # (batch, len, len)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,
                 emb_weight=None):

        super(Decoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        if emb_weight is not None:
            self.tgt_word_emb.weight.data.copy_(emb_weight)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        # Word embedding look up
        dec_input = self.tgt_word_emb(tgt_seq)

        # Position Encoding addition
        dec_input += self.position_enc(tgt_pos)

        # Decode
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)

        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, n_src_vocab, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8,
                 d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
                 dropout=0.1, proj_share_weight=True, embs_share_weight=True,
                 encoder_emb_weight=None, decoder_emb_weight=None):

        super(Transformer, self).__init__()

        self.n_max_seq = n_max_seq

        self.encoder = Encoder(
            n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout,
            emb_weight=encoder_emb_weight)
        self.decoder = Decoder(
            n_tgt_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout,
            emb_weight=decoder_emb_weight)
        self.tgt_word_proj = Linear(d_model, n_tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module output shall be the same.'

        if proj_share_weight:
            # Share the weight matrix between tgt word embedding/projection
            assert d_model == d_word_vec
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight

        if embs_share_weight:
            # Share the weight matrix between src/tgt word embeddings
            # assume the src/tgt word vec size are the same
            assert n_src_vocab == n_tgt_vocab, \
                "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    @staticmethod
    def get_position(seq_tensor, is_decode=False):
        if is_decode:
            pos = [[pos_i + 1 for pos_i, token_i in enumerate(seq)]
                   for seq in seq_tensor]
        else:
            pos = [[pos_i + 1 if token_i != Constants.PAD else 0
                    for pos_i, token_i in enumerate(seq)]
                   for seq in seq_tensor]
        pos = torch.from_numpy(np.array(pos, dtype=np.long))
        pos = autograd.Variable(pos)
        if torch.cuda.is_available():
            pos = pos.cuda()
        return pos

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def _batch_loss(self, criterion, dec_output, gold):
        batch_size = dec_output.size(0)
        loss = 0
        for batch_i, dec in enumerate(dec_output.chunk(batch_size)):
            logit = self.tgt_word_proj(dec).squeeze(0)
            # print(logit)
            # print(gold[batch_i, :])
            loss += criterion(logit, gold[batch_i, :])
        return loss

    def batch_loss(self, criterion, dec_output, gold):
        batch_size = dec_output.size(0)
        # loss = 0
        for batch_i, dec in enumerate(dec_output.chunk(batch_size)):
            logit = self.tgt_word_proj(dec).squeeze(0)
            loss = criterion(logit, gold[batch_i, :])
            yield loss

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)  # (batch, len, embed_dim)
        seq_logit = self.tgt_word_proj(dec_output)

        # return dec_output
        return seq_logit

    def translate(self, src_seq, src_pos):
        ''' Decode by choosing the largest probability '''
        batch_size = src_seq.size(0)
        enc_output, *_ = self.encoder(src_seq, src_pos)

        # init
        dec_seq = torch.LongTensor(batch_size, 1).fill_(Constants.BOS)
        dec_seq = autograd.Variable(dec_seq, volatile=True, requires_grad=False)
        if torch.cuda.is_available():
            dec_seq = dec_seq.cuda()

        # decode
        actives = set([i for i in range(batch_size)])
        for i in range(self.decoder.n_max_seq - 1):
            dec_pos = self.get_position(dec_seq.data, is_decode=True)

            dec_output, *_ = self.decoder(dec_seq, dec_pos, src_seq, enc_output)
            dec_output = dec_output[:, -1, :]  # 僅取最後一個token

            logit = self.tgt_word_proj(dec_output)
            pred = logit.max(1)[1].unsqueeze(1)  # (batch, 1)
            dec_seq = torch.cat([dec_seq, pred], dim=1)

            _actives = set()
            for batch_idx in actives:
                if pred.data[batch_idx, 0] != Constants.EOS:
                    _actives.add(batch_idx)
            actives = _actives

            if len(actives) == 0:
                break

        # decode again to gather grad
        dec_seq = dec_seq.data

        dec_seq = helper.pad_sequence(dec_seq, self.n_max_seq, Constants.PAD)
        dec_seq = autograd.Variable(dec_seq)
        if torch.cuda.is_available():
            dec_seq = dec_seq.cuda()
        dec_pos = self.get_position(dec_seq.data)

        dec_output, *_ = self.decoder(dec_seq, dec_pos, src_seq, enc_output)
        logit = self.tgt_word_proj(dec_output)
        probs = F.softmax(logit)

        return probs
