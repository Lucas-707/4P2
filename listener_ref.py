import torch
import torchaudio
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torchaudio.transforms as tat
import numpy as np
import os
import random
import math

# my pblstm 
class pBLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, downsample=False, dropout=0.05):
        super(pBLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.blstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True, dropout = dropout) # TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size
        self.downsample = downsample

    def forward(self, x_packed): # x_packed is a PackedSequence
        x, lengths = pad_packed_sequence(x_packed, batch_first=True)
        if self.downsample:
            x, lengths = self.trunc_reshape(x, lengths)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.blstm(x)
        return x

    def trunc_reshape(self, x, x_lens):
        if x.shape[1] % 2 != 0:
            x = x[:, :-1, :]
            x_lens -= 1
        B, L, C = x.shape
        x = x.reshape(B, L // 2, C * 2)
        x_lens = x_lens // 2
        return x, x_lens

# ref pblstm
class pBLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, downsample = False):
        super(pBLSTM, self).__init__()
        self.dropout = 0
        self.down = downsample
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.blstm = nn.LSTM(input_size, hidden_size, bidirectional = True, dropout = 0.05)

    def forward(self, x_packed): # x_packed is a PackedSequence
        self.dropout = 0.05
        # TODO: Pad Packed Sequence
        (x_pad, x_len) = pad_packed_sequence(x_packed, batch_first = True)
        x_pad = lockeddropout(self.dropout)(x_pad)
        if self.down:
            (x_reshape, x_len_reshape) = self.trunc_reshape(x_pad, x_len)
        else:
            (x_reshape, x_len_reshape) = (x_pad, x_len)

        final = pack_padded_sequence(x_reshape, x_len_reshape, batch_first = True, enforce_sorted=False)
        value, _ = self.blstm(final)
        return value

    def trunc_reshape(self, x, x_lens):
        # TODO: If you have odd number of timesteps, how can you handle it? (Hint: You can exclude them)
        if x.shape[1] % 2 != 0:
            x = x[:, :-1, :]
            x_lens = x_lens - 1
        # TODO: Reshape x. When reshaping x, you have to reduce number of timesteps by a downsampling factor while increasing number of features by the same factor
        x = x.reshape(x.shape[0], x.shape[1] // 2, x.shape[2] * 2)
        # TODO: Reduce lengths by the same downsampling factor
        x_lens = x_lens // 2
        return x, x_lens

class lockeddropout(nn.Module):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()
    def forward(self, x):
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(x.shape[0], 1, x.shape[2], requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'
    



class Listener(torch.nn.Module):
  def __init__(self, input_size, encoder_hidden_size):
    super(Listener, self).__init__()
    self.dropout = 0
    self.input_size = input_size
    self.encoder_hidden_size = encoder_hidden_size

    #TODO: You can use CNNs as Embedding layer to extract features. Keep in mind the Input dimensions and expected dimension of Pytorch CNN.
    self.embedding1 = nn.Sequential(
        nn.Conv1d(input_size, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
    )
    self.embedding2 = nn.Sequential(
        nn.Conv1d(128 + input_size, 128, kernel_size = 5, stride = 1, padding = 2),
        nn.BatchNorm1d(128),
        nn.ReLU(),
    )
    self.embedding3 = nn.Sequential(
        nn.Conv1d(128, 128, kernel_size = 7, stride = 1, padding = 3),
        nn.BatchNorm1d(128),
        nn.ReLU(),
    )

    self.embedding4 = nn.Sequential(
        nn.Conv1d(128 + 128, encoder_hidden_size // 4, kernel_size = 7, stride = 1, padding = 3),
        nn.BatchNorm1d(encoder_hidden_size // 4),
        nn.ReLU(),
    )

    self.pBLSTMs = torch.nn.Sequential( # How many pBLSTMs are required?
        pBLSTM(encoder_hidden_size // 4, 240, downsample = False),
        pBLSTM(480, 240, downsample = False),
        pBLSTM(960, 240, downsample = True),
        pBLSTM(960, encoder_hidden_size // 2, downsample = True)
    )

  def forward(self, x, x_lens):    
    # Where are x and x_lens coming from? The dataloader
    x = x.permute(0, 2, 1)
    #TODO: Call the embedding layer
    x_embed = self.embedding1(x)
    x_embed = nn.Dropout(self.dropout)(x_embed)
    x_embed2 = self.embedding2(torch.cat((x_embed, x), dim = 1))
    x_embed2 = nn.Dropout(self.dropout)(x_embed2)
    x_embed3 = self.embedding3(x_embed2)
    x_embed3 = nn.Dropout(self.dropout)(x_embed3)
    x_embed4 = self.embedding4(torch.cat((x_embed3, x_embed2), dim = 1))
    x_embed4 = nn.Dropout(self.dropout)(x_embed4)
    # print("after embedding: ", x_embed.shape) # [batch, features, time_length]
    x_embed4 = x_embed4.permute(0, 2, 1)

    x_pack = pack_padded_sequence(x_embed4, x_lens, batch_first = True, enforce_sorted = False)
    final = self.pBLSTMs(x_pack)
    encoder_outputs, encoder_lens = pad_packed_sequence(final, batch_first = True)

    return encoder_outputs, encoder_lens