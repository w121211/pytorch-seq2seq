import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from ape import helper, Constants


class ResBlock(nn.Module):
    def __init__(self, d_model):
        super(ResBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(d_model, d_model, 5, padding=2),  # nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(d_model, d_model, 5, padding=2),  # nn.Linear(DIM, DIM),
        )

    def forward(self, x):
        y = self.res_block(x)
        return x + (0.3 * y)


class BinaryClassifierCNN(nn.Module):
    '''
    分辨seq為真實或機器產生

    Args:

    Inputs: input_var
        - **input_var** (batch, seq_length)

    Outputs: prob
        - **prob** (batch, 1) 輸出[0~1]的值
    '''

    def __init__(self, num_vocab, embed_dim, num_kernel, kernel_sizes, dropout_p):
        super(BinaryClassifierCNN, self).__init__()
        self.min_len = max(kernel_sizes)  # 沒直接用到，但有些training會需要這個參數

        self.embed = nn.Embedding(num_vocab, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_kernel, (k, embed_dim)) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout_p)
        self.conv2out = nn.Linear(len(kernel_sizes) * num_kernel, 1)

    def forward(self, x, input_onehot=False):
        # x = self.check_input(x)  # (batch, seq_len)
        if input_onehot:
            batch_size = x[0].size(0)
            bos = self.embed(autograd.Variable(
                torch.LongTensor([Constants.BOS] * batch_size))).unsqueeze(1)
            x = [token.mm(self.embed.weight).unsqueeze(1) for token in x]
            x.insert(0, bos)
            x = torch.cat(x, dim=1)
        else:
            x = self.embed(x)  # (batch, input_size, embed_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_size, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch, C_out, Width), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch,C_out), ...]*len(Ks)
        x = torch.cat(x, 1)  # (batch, len(kernal_sizes) * kernal_num)
        x = self.dropout(x)  # N x len(Ks)*Co
        logit = self.conv2out(x)  # (batch, class_num)
        logit.squeeze_(1)  # (batch)
        out = F.sigmoid(logit)  # out介於[0-1]，表示prob
        return out
        # return logit

    def check_input(self, x):
        if x.size(1) < self.min_len:
            x = helper.pad_seq(x.data, self.min_len, Constants.PAD)
        return x


class TestDiscriminator(nn.Module):
    def __init__(self, n_vocab, d_model, max_len):
        super(TestDiscriminator, self).__init__()

        self.max_len = max_len
        self.d_model = d_model

        self.vocab2embed = nn.Linear(n_vocab, d_model)
        self.block = nn.Sequential(
            ResBlock(d_model),
            ResBlock(d_model),
            ResBlock(d_model),
            ResBlock(d_model),
            ResBlock(d_model), )
        self.conv2logit = nn.Linear(max_len * d_model, 1)
        # self.conv1d = nn.Conv1d(n_vocab, d_model, 1)
        # self.pool = nn.MaxPool1d()

    def forward(self, x):
        x = self.vocab2embed(x)  # (batch, max_len, n_vocab)
        x = x.transpose(1, 2)  # (batch, n_vocab, max_len)
        x = self.block(x)  # (batch, model_dim, max_len)
        x = x.view(-1, self.max_len * self.d_model)  # (batch, model_dim * seq_len)
        logit = self.conv2logit(x)  # (batch, 1)
        return logit


class ReinforceGenerator(nn.Module):
    """
    增加 GAN generator 所需要的其他功能，最好的方式應為直接更動原始Seq2seq class，不過這樣會影響original repository

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio, volatile
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.
    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(ReinforceGenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function
        self.max_len = self.decoder.max_len

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        decoder_outputs, decoder_hidden, other = self.decoder(
            inputs=target_variable, encoder_hidden=encoder_hidden, encoder_outputs=encoder_outputs,
            function=self.decode_function, teacher_forcing_ratio=teacher_forcing_ratio)

        step_probs = decoder_outputs

        # 忽略掉每個seq的length，直接合併成max_length
        # length = other['length'][0]
        # tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        # tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]

        out_seqs = torch.cat(other[DecoderRNN.KEY_SEQUENCE], dim=1)  # batch x max_length
        out_lengths = other[DecoderRNN.KEY_LENGTH]

        return out_seqs, out_lengths, step_probs

    def rollout(self, input_variable, input_lengths=None, num_rollout=64):
        if input_variable.size(0) != 1:
            raise Exception('rollout 只接受 batch_size = 1, ie 一次一個 real_seq')

        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        inputs, batch_size, max_length = self.decoder._validate_args(
            None, encoder_hidden, encoder_outputs, function=None, teacher_forcing_ratio=0)
        h_n, c_n = self.decoder._init_state(encoder_hidden)

        # stack tensors for batch sampling: (1, *) -> (num_rollout, *)
        inputs = torch.cat([inputs for _ in range(num_rollout)], dim=0)
        h_n = torch.cat([h_n for _ in range(num_rollout)], dim=1)
        c_n = torch.cat([c_n for _ in range(num_rollout)], dim=1)
        encoder_outputs = torch.cat([encoder_outputs for _ in range(num_rollout)], dim=0)
        decoder_hidden = (h_n, c_n)

        # a list of tensor_token (num_rollout x 1)
        seq_symbols, log_probs, entropies = [], [], []
        _indices = list(range(num_rollout))
        decoder_input = inputs[:, 0].unsqueeze(1)

        for i in range(max_length):
            decoder_output, decoder_hidden, step_attn = self.decoder.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, function=F.softmax)
            # step_output = decoder_output.squeeze(1)
            # symbols = decode(di, step_output, step_attn)
            # decoder_input = symbols

            # decode
            probs = decoder_output.squeeze(1)
            symbol = probs.multinomial(1).data  # select an action
            prob = probs[_indices, symbol.squeeze(1).tolist()]  # get the probability of the action
            entropy = -(probs * probs.log()).sum(dim=1)

            seq_symbols.append(symbol)
            log_probs.append(prob.log())
            entropies.append(entropy)

        rollouts = torch.cat(seq_symbols, 1)
        return rollouts, seq_symbols, log_probs, entropies

    def batch_loss(self, step_probs, target_variable=None, criterion=nn.NLLLoss()):
        if target_variable is None:
            raise ValueError('需要target_variable才能計算loss')

        target = self._validate_variables(target=target_variable)

        batch_size = target_variable.size(0)

        loss = 0
        for step, step_prob in enumerate(step_probs):
            loss = criterion(step_prob.contiguous().view(batch_size, -1), target[:, step])
            loss /= batch_size
            yield loss

    def _validate_variables(self, target=None):
        if target is not None:
            # 第一欄中有<sos>? -> 去掉第一欄
            if target[:, 0].eq(self.decoder.sos_id).sum().data[0] > 0:
                target = target[:, 1:]

                # 若len未達到max_len -> padding
                # max_len = self.decoder.max_length
                # if target.size(1) < max_len:
                #     target = helper.pad_sequence(target, max_len, seq2seq.pad_id)
        return target


class CycleGAN(nn.Module):
    def __init__(self, g_a, g_b, d_a, d_b):
        super(CycleGAN, self).__init__()
        self.g_a = g_a
        self.g_b = g_b
        self.d_a = d_a
        self.d_b = d_b

    def flatten_parameters(self):
        self.g_a.flatten_parameters()
        self.g_b.flatten_parameters()

    def pg_loss(self, log_probs, entropies, rewards):
        ''' 給予1個real_seq，用此執行REINFORCE algorithm
        步驟：
        1. 輸入 real_seq
        2. 用seq2seq model生成數個samples
        3. 用 discriminator 分辨每個 sample 的真實性，輸出probability，此prob為每個sample的final_reward
        4. intermediate_reward (每個action的reward) = log(prob_of_action) * final_reward
        5. normalization
        6. loss = -(sum of all intermediate_rewards)

        Args:
            model   a generator model
            dis     a discriminator model
            input_var   a real sequence from source
        '''
        loss = 0
        for log_prob, entropy in zip(log_probs, entropies):
            loss += -(log_prob * rewards).sum()

        # normalize
        num_rollout = rewards.size(0)
        loss /= len(log_probs) * num_rollout

        return loss
