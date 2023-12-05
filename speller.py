import torch
import torchaudio
from torch import nn, Tensor
import random
import torch.nn.functional as F
from attention import Attention
from utils import PermuteBlock
DEVICE = 'cuda:0'
VOCAB = [
    '<pad>', '<sos>', '<eos>',
    'A',   'B',    'C',    'D',
    'E',   'F',    'G',    'H',
    'I',   'J',    'K',    'L',
    'M',   'N',    'O',    'P',
    'Q',   'R',    'S',    'T',
    'U',   'V',    'W',    'X',
    'Y',   'Z',    "'",    ' ',
]

VOCAB_MAP = {VOCAB[i]:i for i in range(0, len(VOCAB))}

PAD_TOKEN = VOCAB_MAP["<pad>"]
SOS_TOKEN = VOCAB_MAP["<sos>"]
EOS_TOKEN = VOCAB_MAP["<eos>"]


# locked dropout for cell
class LockedDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()
        
    def forward(self, x, timestep):
        if not self.training or not self.p:
            return x
        if timestep == 0:
            x = x.clone()
            mask = x.new_empty(x.shape, requires_grad=False).bernoulli_(1 - self.p)
            mask = mask.div_(1 - self.p)
            self.mask = mask
        return x * self.mask

class EmbeddingDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()
        
    def forward(self, embed, words):
        if self.training:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - self.p).expand_as(embed.weight) / (1 - self.p)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight
        X = torch.nn.functional.embedding(words, masked_embed_weight)
        return X

class Speller(torch.nn.Module):

  # Refer to your HW4P1 implementation for help with setting up the language model.
  # The only thing you need to implement on top of your HW4P1 model is the attention module and teacher forcing.

  def __init__(self, attender:Attention, embedding_size, speller_hidden_size, projection_size, max_timesteps, vocab_size=31):
    super(). __init__()
    self.attend = attender # Attention object in speller
    self.max_timesteps = max_timesteps# Max timesteps
    self.embedding_size = embedding_size
    self.speller_hidden_size = speller_hidden_size
    self.projection_size = projection_size

    self.embedding_layer =  torch.nn.Embedding(vocab_size, embedding_size)# Embedding layer to convert token to latent space
    self.embedding = EmbeddingDropout(0.1)

    self.lstm_cells =  torch.nn.Sequential(
        nn.LSTMCell(embedding_size + projection_size, speller_hidden_size),
        nn.LSTMCell(speller_hidden_size, speller_hidden_size),
        nn.LSTMCell(speller_hidden_size, speller_hidden_size)
    )# Create a sequence of LSTM Cells
    self.lock_drops = torch.nn.Sequential(
            LockedDropout(0.2),
            LockedDropout(0.2),
            LockedDropout(0.2),
    )
    
    # For CDN (Feel free to change)
    self.output_to_char = nn.Linear(projection_size + speller_hidden_size, embedding_size)# Linear module to convert outputs to correct hidden size (Optional: TO make dimensions match)
    self.activation = nn.Tanh() # Check which activation is suggested
    self.char_prob = nn.Linear(embedding_size, vocab_size) # Linear layer to convert hidden space back to logits for token classification
    self.char_prob.weight = self.embedding_layer.weight # Weight tying

  def lstm_step(self, input_word, hidden_state, t):
    for i in range(len(self.lstm_cells)):
      if i==0:
        hidden_state[i] = self.lstm_cells[i](input_word, hidden_state[i])
        hidden_state[i] = (self.lock_drops[i](hidden_state[i][0], t), hidden_state[i][1])
      else:
        hidden_state[i] = self.lstm_cells[i](hidden_state[i-1][0], hidden_state[i])
        hidden_state[i] = (self.lock_drops[i](hidden_state[i][0], t), hidden_state[i][1])  
    return hidden_state[-1][0], hidden_state # What information does forward() need?

  def CDN(self, input):
    out = self.output_to_char(input)
    out = self.activation(out)
    out = self.char_prob(out)
    return out
  
  def forward (self, y=None, teacher_forcing_ratio=1, batch_size=1):
    attn_context = torch.zeros(batch_size, self.projection_size).to(DEVICE) # initial context tensor for time t = 0
    output_symbol = torch.ones((batch_size, SOS_TOKEN), dtype=torch.int64).to(DEVICE) # Set it to SOS for time t = 0
    raw_outputs = []
    attention_plot = []

    if y is None:
      timesteps = self.max_timesteps
      teacher_forcing_ratio = 0 #Why does it become zero?
    else:
      timesteps = y.shape[1] # How many timesteps are we predicting for?

    hidden_states_list = [None] * len(self.lstm_cells) # Initialize your hidden_states list here similar to HW4P1
    for t in range(timesteps):
      p = random.random() # generate a probability p between 0 and 1
      if y is not None and t>0:
        if p < teacher_forcing_ratio: # Why do we consider cases only when t > 0? What is considered when t == 0? Think.
          output_symbol = y[:, t-1] # Take from y, else draw from probability distribution
        else:
          output_symbol = nn.functional.gumbel_softmax(raw_outputs[t - 1])

      char_embed = self.embedding(self.embedding_layer, output_symbol).squeeze() # Embed the character symbol
      # Concatenate the character embedding and context from attention, as shown in the diagram
      lstm_input = torch.cat((char_embed, attn_context), dim=1)
      lstm_out, hidden_states_list = self.lstm_step(lstm_input, hidden_states_list, t) # Feed the input through LSTM Cells and attention.
      if torch.any(torch.isnan(lstm_out)):
            print(f"speller lstm out is nan! time={t} lstm_out: ", lstm_out)
            assert(False)
      # What should we retrieve from forward_step to prepare for the next timestep?
      attn_context, attn_weights = self.attend.compute_context(lstm_out) # Feed the resulting hidden state into attention
      if torch.any(torch.isnan(attn_context)):
            print(f"attn_context is nan! time={t} attn_context: ", attn_context)
            assert(False)
      cdn_input = torch.cat((lstm_out, attn_context), dim=1) # TODO: You need to concatenate the context from the attention module with the LSTM output hidden state, as shown in the diagram
      raw_pred = self.CDN(cdn_input) # call CDN with cdn_input
      if torch.any(torch.isnan(raw_pred)):
            print(f"CDN is nan! time={t} CDN: ", raw_pred)
            assert(False)
      # Generate a prediction for this timestep and collect it in output_symbols
      output_symbol = torch.argmax(raw_pred, dim=1) # Draw correctly from raw_pred
      raw_outputs.append(raw_pred) # for loss calculation
      attention_plot.append(attn_weights) # for plotting attention plot

    attention_plot = torch.stack(attention_plot, dim=1)
    raw_outputs = torch.stack(raw_outputs, dim=1)
    return raw_outputs, attention_plot