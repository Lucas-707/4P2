import torch
import torchaudio
from torch import nn, Tensor
import torch.nn.functional as F
from utils import PermuteBlock

class Attention(torch.nn.Module):
  def __init__(self, listener_hidden_size, speller_hidden_size, projection_size):
    super().__init__()
    self.KW = nn.Linear(listener_hidden_size, projection_size)
    self.VW = nn.Linear(listener_hidden_size, projection_size)
    self.QW = nn.Linear(speller_hidden_size, projection_size)
    self.permute    = PermuteBlock()

  def set_key_value(self, encoder_outputs):
    self.key = self.KW(encoder_outputs)
    self.value = self.VW(encoder_outputs)

  def compute_context(self, decoder_context):
    # compute the key, query and value
    query   = self.QW(decoder_context) # TODO
    # print(self.key.shape, query.shape)
    raw_weights = torch.bmm(torch.unsqueeze(query, 1), self.permute(self.key)).squeeze(1)
    attention_weights = torch.nn.functional.softmax(raw_weights, dim=1)
    attention_context = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1)
    return attention_context, attention_weights
  
# class Attention(torch.nn.Module):
#   def __init__(self, listener_hidden_size, speller_hidden_size, projection_size):
#     super().__init__()
#     self.VW = nn.Linear(listener_hidden_size, projection_size)
#     self.KW = nn.Linear(listener_hidden_size, projection_size)
#     self.QW = nn.Linear(speller_hidden_size, projection_size)
#     # self.batch_size = config['batch_size']

#   def set_key_value(self, encoder_outputs):
#     self.key = self.KW(encoder_outputs)
#     self.value = self.VW(encoder_outputs)

#   def compute_context(self, decoder_context):
#     query = self.QW(decoder_context) # B*P
#     raw_weights = torch.bmm(torch.unsqueeze(query, 1), self.key.transpose(1, 2)).squeeze(1)
#     attention_weights = torch.nn.functional.softmax(raw_weights, dim=1)
#     context = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1)
#     return context, attention_weights