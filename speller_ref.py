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

class Speller(torch.nn.Module):

  # Refer to your HW4P1 implementation for help with setting up the language model.
  # The only thing you need to implement on top of your HW4P1 model is the attention module and teacher forcing.

  def __init__(self, attender:Attention, embedding_size, speller_hidden_size, projection_size, max_timesteps, vocab_size=31):
    super(). __init__()

    self.attend = attender # Attention object in speller
    self.max_timesteps = max_timesteps
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.speller_hidden_size = speller_hidden_size
    self.projection_size = projection_size

    self.embedding = torch.nn.Embedding(vocab_size, embedding_size) # Embedding layer to convert token to latent space
    self.dropout = torch.nn.Dropout(0.75)
    self.lstm_cells = torch.nn.Sequential(
        torch.nn.LSTMCell(embedding_size + projection_size, speller_hidden_size),
        torch.nn.LSTMCell(speller_hidden_size, speller_hidden_size),
        torch.nn.LSTMCell(speller_hidden_size, speller_hidden_size)
    ) # Create a sequence of LSTM Cells

    # For CDN (Feel free to change)
    self.output_to_char = torch.nn.Linear(projection_size + speller_hidden_size, embedding_size)# Linear module to convert outputs to correct hidden size (Optional: TO make dimensions match)
    self.activation = torch.nn.Tanh() # Check which activation is suggested
    self.char_prob = torch.nn.Linear(embedding_size, vocab_size) # Linear layer to convert hidden space back to logits for token classification
    self.char_prob.weight = self.embedding.weight # Weight tying (From embedding layer)


  def lstm_step(self, input_word, hidden_state):

    for i in range(len(self.lstm_cells)):
      if i == 0:
        hidden_state[i] = self.lstm_cells[i](input_word, hidden_state[i])
      else:
        hidden_state[i] = self.lstm_cells[i](hidden_state[i - 1][0], hidden_state[i])
    return hidden_state[-1][0], hidden_state # What information does forward() need?

  def CDN(self, x):
    # Make the CDN here, you can add the output-to-char
    x = self.output_to_char(x)
    x = self.activation(x)
    x = self.char_prob(x)
    return x

  def forward(self, y=None, teacher_forcing_ratio=1, batch_size = 1):

    attn_context = torch.zeros(batch_size, self.projection_size).to(DEVICE) # initial context tensor for time t = 0
    output_symbol = torch.ones((batch_size, SOS_TOKEN), dtype=torch.int64).to(DEVICE) # Set it to SOS for time t = 0
    raw_outputs = []
    attention_plot = []

    if y is None:
      timesteps = self.max_timesteps
      teacher_forcing_ratio = 0 # Why does it become zero?
    else:
      timesteps = y.shape[1] # How many timesteps are we predicting for?

    hidden_states_list = [None] * len(self.lstm_cells)# Initialize your hidden_states list here similar to HW4P1

    for t in range(timesteps):
      p = random.random()
      assert (p >= 0 and p < 1)

      if y is not None:
        if p < teacher_forcing_ratio and t > 0: # Why do we consider cases only when t > 0? What is considered when t == 0? Think.
          output_symbol = y[:, t - 1] # Take from y, else draw from probability distribution
        elif p >= teacher_forcing_ratio and t > 0:
          output_symbol = torch.nn.functional.gumbel_softmax(raw_outputs[t - 1])


      char_embed = self.dropout(self.embedding(output_symbol).squeeze()) # Embed the character symbol

      # Concatenate the character embedding and context from attention, as shown in the diagram
      # print(char_embed.shape, attn_context.shape) # [128, 256], [128, 128]
      lstm_input = torch.cat((char_embed, attn_context), dim=1)

      rnn_out, hidden_states_list = self.lstm_step(lstm_input, hidden_states_list) # Feed the input through LSTM Cells and attention.
      # What should we retrieve from forward_step to prepare for the next timestep?
      attn_context, attn_weights = self.attend.compute_context(rnn_out)

      # attn_context, attn_weights = self.attend.compute_context(...) # Feed the resulting hidden state into attention

      # print(rnn_out.shape, len(hidden_states_list), attn_context.shape, attn_weights.shape, attn_weights)
      # torch.Size([128, 512]) 3 torch.Size([128, 1, 128]) torch.Size([128, 1, 734])
      cdn_input = torch.cat((rnn_out, attn_context), dim=1) # TODO: You need to concatenate the context from the attention module with the LSTM output hidden state, as shown in the diagram

      raw_pred = self.CDN(cdn_input) # call CDN with cdn_input

      # Generate a prediction for this timestep and collect it in output_symbols
      output_symbol = torch.argmax(raw_pred, dim=1) # Draw correctly from raw_pred

      raw_outputs.append(raw_pred) # for loss calculation
      attention_plot.append(attn_weights) # for plotting attention plot


    attention_plot = torch.stack(attention_plot, dim=1)
    raw_outputs = torch.stack(raw_outputs, dim=1)

    return raw_outputs, attention_plot
