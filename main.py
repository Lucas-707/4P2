import torch
import torchaudio
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torchaudio.transforms as tat

import numpy as np
import os
import random
import gc
import time

import pandas as pd
from tqdm.notebook import tqdm as blue_tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
import json
import math
from typing import Optional, List

#imports for decoding and distance calculation
try:
    import wandb
    import torchsummaryX
    import Levenshtein
except:
    print("Didnt install some/all imports")
from torchsummaryX import summary
import warnings
warnings.filterwarnings('ignore')

from ASR import ASRModel
from utils import plot_attention, save_model
from experiment import train, validate

DEVICE = 'cuda:1'
print("Device: ", DEVICE)
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


config = dict (
    train_dataset       = 'train-clean-100', # train-clean-100, train-clean-360, train-clean-460
    batch_size          = 128,
    epochs              = 100,
    lr                  = 1e-3,
    weight_decay        = 5e-5
)

from dataset import SpeechDatasetME, SpeechDatasetTest

DATA_DIR        = 'data/11-785-f23-hw4p2'
PARTITION       =config['train_dataset']
CEPSTRAL        = True
train_dataset   = SpeechDatasetME( # Or AudioDatasetME
    root        = DATA_DIR,
    partition   = PARTITION,
    cepstral    = CEPSTRAL
)
valid_dataset   = SpeechDatasetME(
    root        = DATA_DIR,
    partition   = 'dev-clean',
    cepstral    = CEPSTRAL
)
test_dataset    = SpeechDatasetTest(
    root        = DATA_DIR,
    partition   = 'test-clean',
    cepstral    = CEPSTRAL,
)
gc.collect()
train_loader    = torch.utils.data.DataLoader(
    dataset     = train_dataset,
    batch_size  = config['batch_size'],
    shuffle     = True,
    num_workers = 4,
    pin_memory  = True,
    collate_fn  = train_dataset.collate_fn
)
valid_loader    = torch.utils.data.DataLoader(
    dataset     = valid_dataset,
    batch_size  = config['batch_size'],
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True,
    collate_fn  = valid_dataset.collate_fn
)
test_loader     = torch.utils.data.DataLoader(
    dataset     = test_dataset,
    batch_size  = config['batch_size'],
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True,
    collate_fn  = test_dataset.collate_fn
)

model = ASRModel(
    input_size=28,
    listener_hidden_size=512,
    speller_hidden_size=512,
    projection_size=256,
    embedding_size=256,
    max_timesteps=550,
)
model = model.to(DEVICE)
for data in train_loader:
    x, y, lx, ly = data
    print("x,y, lx, ly:", x.shape, y.shape, lx.shape, ly.shape)
    break
summary(model, x.to(DEVICE), lx, y.to(DEVICE))

optimizer   = torch.optim.AdamW(model.parameters(), lr= config['lr'], weight_decay=config['weight_decay'])
criterion   = torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=PAD_TOKEN)
# scaler      = torch.cuda.amp.GradScaler()
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], 1e-6)

# This is for checkpointing, if you're doing it over multiple sessions
last_epoch_completed = 0
start = last_epoch_completed
end = config["epochs"]
best_lev_dist = float("inf") # if you're restarting from some checkpoint, use what you saw there.
# epoch_model_path = #TODO set the model path( Optional, you can just store best one. Make sure to make the changes below )
best_model_path = "/home/yuwu3/4P2/checkpoints/checkpoint.pt"#TODO set best model path
wandb.login(key="e40d3c88212e0edbeeb32f1487a248a7c144260c")
run = wandb.init(
    name = "ASR_ref", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "hw4p2-ablations", ### Project should be created in your wandb account
    config = config ### Wandb Config for your run
)

best_lev_dist = float("inf")
tf_rate = 1.0

for epoch in range(0, config['epochs']):

    print("\nEpoch: {}/{}".format(epoch+1, config['epochs']))
    curr_lr = float(optimizer.param_groups[0]['lr'])
    # Call train and validate, get attention weights from training
    teacher_forcing_ratio = 1
    train_loss, running_perplexity, attention_plot = train(model, train_loader, criterion, optimizer, teacher_forcing_ratio, DEVICE)
    valid_loss, valid_dist = validate(model, valid_loader, DEVICE)
    # Print your metrics
    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Dist {:.04f}%\t Val Loss {:.04f}".format(valid_dist, valid_loss))
    # Plot Attention for a single item in the batch
    plot_attention(attention_plot[0].cpu().detach().numpy(), "./attn_heat", epoch)
    
    # Log metrics to Wandb
    wandb.log({
        'train_loss': train_loss,
        'valid_dist': valid_dist,
        'valid_loss': valid_loss,
        'lr'        : curr_lr
    })
    # Optional: Scheduler Step / Teacher Force Schedule Step
    scheduler.step()

    if valid_dist <= best_lev_dist:
        best_lev_dist = valid_dist
        save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
        # wandb.save(best_model_path)
        print("Saved best model")

run.finish()
