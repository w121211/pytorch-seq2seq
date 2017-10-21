import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .DecoderRNN import DecoderRNN



class ReinforceDecoderRNN(DecoderRNN):
    def select_action(self):
        pass

    def sample(self, encorder_outputs, encorde_hiddenr, sample_size=128, functionn=F.log_softmax):
        '''
        給予`encoded content`，生成samples
        :param encorder_outputs:
        :param encorde_hiddenr:
        :param sample_size:
        :param functionn:
        :return:
        '''
        # samples = torch.LongTensor(sample_size)  # (sample_size, max_seq_len)
        samples = []

        for i in range(max_length):
            for s in samples:
                input_var = s[:, -1]
                prob, hidden = self.forward(input_var, hidden)
                action = prob.multinomial(1)
                self.past_actions += action
                s.expand(action)
        return samples

    def forward(self, inputs=None, encoder_hidden=None, function=F.log_softmax,
                encoder_outputs=None, teacher_forcing_ratio=0):
        '''
        batch_size僅能為1

        :param inputs: (batch_size, seq_len) 可設為target seq，用於teacher forcing
                    eg, [[sos_id, token_a_id, token_b_id, ...],
                         [sos_id, token_c_id, token_d_id, ...]]
        :param encoder_hidden:
        :param function:
        :param encoder_outputs: (batch, seq_len, hidden_size * num_directions)
        :param teacher_forcing_ratio:
        :return: decoder_outputs: (batch_size, sample_size, output_seq_max_len)
                    eg, batch_id = 1
                        sample_1  [[sos_id, token_a_id, token_b_id, ...]
                        sample_2   [sos_id, token_c_id, token_d_id, ...]
                          ...
                        sample_n   [...]]
        '''
        ret_dict = dict()

        if encoder_outputs.size(0) != 1:
            raise ValueError("Batch_size needs to be 1.")

        batch_size = 1
        if inputs is None:
            inputs = Variable(torch.LongTensor([self.sos_id]),
                              volatile=True).view(batch_size, -1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        decoder_hidden = self._init_state(encoder_hidden)

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        samples = torch.LongTensor(batch_size, sample_size, max_length)
        samples.fill(pad_id)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > di) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        decoder_input = inputs[:, 0].unsqueeze(1)  # (batch * sample_size, 1)

        for di in range(max_length):
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                          encoder_outputs,
                                                                          function=function)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict


    def select_action(self):
        pass
