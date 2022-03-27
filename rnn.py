
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

use_cuda = torch.cuda.is_available()

if use_cuda:
    available_device = torch.device('cuda')
else:
    available_device = torch.device('cpu')

# Generic sequential encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, recurrent_unit, n_layers=1, max_length=30, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn_type = recurrent_unit
        self.max_length = max_length
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(self.dropout_p)

        if recurrent_unit == "SRN":
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=n_layers, dropout=self.dropout_p)
        elif recurrent_unit == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, dropout=self.dropout_p)
        elif recurrent_unit == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=self.dropout_p)

    # Creates the initial hidden state
    def initHidden(self, recurrent_unit, batch_size):
        if recurrent_unit == "SRN" or recurrent_unit == "GRU":
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device=available_device)
        elif recurrent_unit == "LSTM":
            hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device=available_device), torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device=available_device))

        return hidden

    # For succesively generating each new output and hidden layer
    def forward(self, training_pair):

        input_variable = training_pair[0]
        target_variable = training_pair[1]

        batch_size = training_pair[0].size()[1]

        hidden = self.initHidden(self.rnn_type, batch_size)

        input_length = input_variable.size()[0]

        for ei in range(input_length):
            emb = self.embedding(input_variable[ei]).unsqueeze(0)
            emb = self.dropout(emb)
            output, hidden = self.rnn(emb, hidden)

        return output, hidden

# Generic sequential decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, recurrent_unit, n_layers=1, dropout_p=0.1, max_length=30, sos_token=None):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(self.hidden_size, self.hidden_size, num_layers=n_layers, dropout=self.dropout_p)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers=n_layers, dropout=self.dropout_p)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=n_layers, dropout=self.dropout_p)

        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.recurrent_unit = recurrent_unit
        self.sos_token = sos_token

    # Perform one step of the forward pass
    def forward_step(self, input, hidden, input_variable):
        emb = self.embedding(input).unsqueeze(0)
        emb = self.dropout(emb)

        output, hidden = self.rnn(emb, hidden)

        output = self.out(output[0])
        return output, hidden

    # Perform the full forward pass
    def forward(self, hidden, training_pair, tf_ratio=0.5):
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        batch_size = training_pair[0].size()[1]

        decoder_hidden = hidden
        decoder_outputs = []

        use_tf = True if random.random() < tf_ratio else False

        if use_tf: # Using teacher forcing
            for di in range(target_variable.size()[0]):
                decoder_input = target_variable[di]
                decoder_output, decoder_hidden = self.forward_step(
                                decoder_input, decoder_hidden, input_variable)
                decoder_outputs.append(decoder_output)

        else: # Not using teacher forcing

            if not self.training:
                output_length = self.max_length
            else:
                output_length = target_variable.size()[0]

            decoder_input = torch.LongTensor([self.sos_token] * batch_size)
            for di in range(output_length):
                decoder_output, decoder_hidden = self.forward_step(
                            decoder_input, decoder_hidden, input_variable) 

                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.view(-1).to(device=available_device)

                decoder_outputs.append(decoder_output)


        return decoder_outputs 


class RNNSeq2Seq(nn.Module):
    def __init__(self, hidden_size, output_size, recurrent_unit, n_layers=1, dropout_p=0.1, max_length=30, sos_token=None):
        super(RNNSeq2Seq, self).__init__()
        
        self.encoder = EncoderRNN(output_size, hidden_size, recurrent_unit, n_layers=n_layers, dropout_p=dropout_p, max_length=max_length)
        self.decoder = DecoderRNN(hidden_size, output_size, recurrent_unit, n_layers=n_layers, dropout_p=dropout_p, max_length=max_length, sos_token=sos_token)

    def forward(self, inp, target):

        inp = inp.transpose(0,1)
        target = target.transpose(0,1)
        _, hidden = self.encoder([inp, target])
        decoder_outputs = self.decoder(hidden, [inp, target], tf_ratio=1.0)
       
        full_decoder_outputs = decoder_outputs[0].unsqueeze(0)
        for outp in decoder_outputs[1:]:
            full_decoder_outputs = torch.cat((full_decoder_outputs, outp.unsqueeze(0)), 0)

        return full_decoder_outputs.transpose(0,1)

    def greedy_inference(model, src, sos_idx, eos_idx, max_length, device):
        model.eval()
        inp = src.transpose(0,1)
        target = None

        _, hidden = model.encoder([inp, target])
        decoder_outputs = model.decoder(hidden, [inp, target], tf_ratio=0.0)

        full_decoder_outputs = decoder_outputs[0].unsqueeze(0)
        for outp in decoder_outputs[1:]:
            full_decoder_outputs = torch.cat((full_decoder_outputs, outp.unsqueeze(0)), 0)

        topv, topi = full_decoder_outputs.transpose(0,1).topk(1)
        preds = topi.squeeze(2)

        soses = torch.ones(src.shape[0], 1).fill_(sos_idx).type_as(src).to(device)
        preds = torch.cat((soses, preds), dim=1) 

        return preds



