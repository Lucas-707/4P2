import torch
import torchaudio
from torch import nn, Tensor
import random
import torch.nn.functional as F
from attention import Attention
from utils import PermuteBlock
import torchaudio.transforms as tat

from listener import TransformerListener
from listener_ref import Listener
# from speller import Speller
from speller_ref import Speller


class ASRModel(torch.nn.Module):
  def __init__(self, input_size, listener_hidden_size, speller_hidden_size, projection_size, 
               embedding_size, max_timesteps): # add parameters
    super().__init__()

    # Pass the right parameters here
    # self.listener = TransformerListener(input_size, listener_hidden_size=listener_hidden_size)
    self.listener = Listener(input_size, listener_hidden_size)
    self.attend = Attention(listener_hidden_size, speller_hidden_size, projection_size)
    self.speller = Speller(self.attend, embedding_size, speller_hidden_size, projection_size, max_timesteps)

    self.augmentations  = torch.nn.Sequential(
            PermuteBlock(),
            tat.TimeMasking(0.2),
            tat.FrequencyMasking(0.2),
            PermuteBlock()
        )
  def forward(self, x,lx,y=None,teacher_forcing_ratio=1):
    if self.training:
      x = self.augmentations(x)
      
    # Encode speech features
    encoder_outputs, _ = self.listener(x,lx)

    # We want to compute keys and values ahead of the decoding step, as they are constant for all timesteps
    # Set keys and values using the encoder outputs
    self.attend.set_key_value(encoder_outputs)

    # Decode text with the speller using context from the attention
    raw_outputs, attention_plots = self.speller(y=y,teacher_forcing_ratio=teacher_forcing_ratio, batch_size=x.shape[0])

    return raw_outputs, attention_plots