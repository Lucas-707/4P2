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

from utils import PermuteBlock

# my position
class PositionalEncoding(torch.nn.Module):
    def __init__(self, projection_size, max_seq_len= 550):
        super().__init__()
        # Read the Attention Is All You Need paper to learn how to code code the positional encoding
        encoding = torch.zeros(max_seq_len, projection_size)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, projection_size, 2).float() * -(np.log(10000.0) / projection_size))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0)
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        batch_encoding = self.encoding.repeat(x.shape[0], 1,1)
        # print(x.shape, batch_encoding.shape)
        return torch.cat([x, batch_encoding[:,:x.shape[1],:]], dim=2)
    
# position from recitation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x


# my pblstm
class pBLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, downsample=False, dropout=0.0):
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

#my transformer block
class TransformerEncoder(torch.nn.Module):
    def __init__(self, projection_size, num_heads, dropout= 0.1):
        super().__init__()
        self.projection_size = projection_size
        # create the key, query and value weights
        self.KW         = nn.Linear(projection_size, projection_size) # TODO
        self.VW         = nn.Linear(projection_size, projection_size) # TODO
        self.QW         = nn.Linear(projection_size, projection_size) # TODO
        self.permute    = PermuteBlock()
        # Compute multihead attention. You are free to use the version provided by pytorch
        self.attention  = nn.MultiheadAttention(embed_dim=projection_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.bn1        = nn.LayerNorm(projection_size, eps=1e-6)  
        self.bn2        = nn.LayerNorm(projection_size, eps=1e-6) 
        # Feed forward neural network
        self.MLP        = nn.Sequential(
            nn.Linear(projection_size, 2 * projection_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * projection_size, projection_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # compute the key, query and value
        key     = self.KW(x)# TODO
        value   = self.VW(x) # TODO
        query   = self.QW(x) # TODO

        # compute the output of the attention module
        out1, _  =  self.attention(query, key, value) # TODO
        if torch.any(torch.isnan(out1)):
            print("self attention out1 is nan! output: ", out1)
            assert(False)
        # Create a residual connection between the input and the output of the attention module
        out1    = out1 + x # TODO
        # Apply batch norm to out1
        out1    = self.bn1(out1) # TODO
        if torch.any(torch.isnan(out1)):
            print("layernorm out1 is nan! output: ", out1)
            assert(False)
        # Apply the output of the feed forward network
        out2    = self.MLP(out1) # TODO
        # Apply a residual connection between the input and output of the  FFN
        out2    = out2 + out1 # TODO
        # Apply batch norm to the output
        out2    = self.bn2(out2) # TODO
        if torch.any(torch.isnan(out2)):
            print("layernorm out2 is nan! output: ", out2)
            assert(False)
        return out2


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "Invalid number of heads or d_model dimensions"
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.wo(out)
        return out
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out
    
class AddNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
    def forward(self, x, residual):
        out = x + residual
        out = self.norm(out)
        return out

# transformer from recitation
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = AddNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = AddNorm(d_model)
    def forward(self, x, mask=None):
        x1 = self.self_attn(x, x, x, mask)
        x = self.norm1(x, x1)
        x1 = self.ffn(x)
        x = self.norm2(x, x1)
        return x
    
class TransformerListener(torch.nn.Module):
    def __init__(self,
                 input_size,
                 base_lstm_layers        = 2,
                 pblstm_layers           = 2,
                 listener_hidden_size    = 256,
                 position_size           = 32,
                 n_heads                 = 8,
                 tf_blocks               = 4,
                 conv_dropout            = 0.0,
                 attn_dropout            = 0.1):
        super().__init__()
        # create an lstm layer
        self.lstm_layers      = nn.Sequential(
            pBLSTM(listener_hidden_size * 2, listener_hidden_size, downsample=True, dropout=0.1),
            pBLSTM(listener_hidden_size * 2, listener_hidden_size, downsample=False, dropout=0.1),
            pBLSTM(listener_hidden_size * 2, listener_hidden_size // 2, downsample=False, dropout=0.1),
        )
        self.permute = PermuteBlock()
        self.embedding2      = nn.Sequential(
            self.permute,
            nn.Conv1d(input_size, input_size*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(input_size*8),
            nn.GELU(),
            # torch.nn.Dropout(conv_dropout),
            nn.Conv1d(input_size*8, input_size*16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(input_size*16),
            nn.GELU(),
            # torch.nn.Dropout(conv_dropout),
            nn.Conv1d(input_size*16, listener_hidden_size, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(listener_hidden_size),
            nn.GELU(),
            # torch.nn.Dropout(conv_dropout),
            self.permute
        )

        # compute the postion encoding
        self.positional_encoding    = PositionalEncoding(listener_hidden_size, max_seq_len=550, dropout=0.1) # TODO

        # create a sequence of transformer blocks
        # self.transformer_encoder    = torch.nn.Sequential(
        #     TransformerEncoder(listener_hidden_size, num_heads=n_heads, dropout=attn_dropout),
        #     TransformerEncoder(listener_hidden_size, num_heads=n_heads, dropout=attn_dropout),
        #     TransformerEncoder(listener_hidden_size, num_heads=n_heads, dropout=attn_dropout),
        #     TransformerEncoder(listener_hidden_size, num_heads=n_heads, dropout=attn_dropout)
        # )
        self.transformer_encoder    = torch.nn.Sequential(
            EncoderBlock(listener_hidden_size, n_heads, listener_hidden_size*2, attn_dropout),
            EncoderBlock(listener_hidden_size, n_heads, listener_hidden_size*2, attn_dropout),
            EncoderBlock(listener_hidden_size, n_heads, listener_hidden_size*2, attn_dropout),
            EncoderBlock(listener_hidden_size, n_heads, listener_hidden_size*2, attn_dropout),
        )
        # for i in range(tf_blocks):
        #     # TODO

    def forward(self, x, x_len):
        # Pass the output through the embedding
        x                  = self.embedding2(x)
        x_len = (x_len - 1) // 4 + 1
        # pack the inputs before passing them to the LSTm
        x_packed                = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        # Pass the packed sequence through the lstm
        lstm_out                = self.lstm_layers(x_packed)
        # lstm_out = x_packed
        # Unpack the output of the lstm
        output, output_lengths  = pad_packed_sequence(lstm_out, batch_first=True)
        
        
        
        # calculate the new output length
        # output_lengths          = (output_lengths - 1) // 4 + 1# TODO
        # calculate the position encoding and concat it to the output
        output  = self.positional_encoding(output) # TODO
        if torch.any(torch.isnan(output)):
            print("listener postional out is nan! output: ", output)
            assert(False)
        # Pass the output of the positional encoding through the transformer encoder
        output  = self.transformer_encoder(output) # TODO
        return output, output_lengths