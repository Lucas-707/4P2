import torch
import torchaudio
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import os
import random

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

def transcript_to_ind(transcript):
    map_func = np.vectorize(VOCAB_MAP.get)
    index_array = map_func(transcript)
    return index_array

def cepstral_mean_normalization(features):
    mean = np.mean(features, axis=0)
    normalized_features = features - mean
    return normalized_features

class SpeechDatasetME(torch.utils.data.Dataset): # Memory efficient
    # Loades the data in get item to save RAM
    def __init__(self, root, partition= "train-clean-360", transforms = None, cepstral=True):
        self.VOCAB      = VOCAB
        self.cepstral   = cepstral

        if partition == "train-clean-100" or partition == "train-clean-360" or partition == "dev-clean":
            mfcc_dir       = f"/home/yuwu3/4P2/data/11-785-f23-hw4p2/{partition}/mfcc"#  path to the mfccs
            transcript_dir = f"/home/yuwu3/4P2/data/11-785-f23-hw4p2/{partition}/transcripts" # path to the transcripts
            mfcc_files          =  sorted(os.listdir(mfcc_dir))# create a list of paths for all the mfccs in the mfcc directory
            transcript_files    =  sorted(os.listdir(transcript_dir))# create a list of paths for all the transcripts in the transcript directory

        else:
            mfcc_dir       = f"/home/yuwu3/4P2/data/11-785-f23-hw4p2/train-clean-100/mfcc"# path to the mfccs in the train clean 100 partition
            transcript_dir = f"/home/yuwu3/4P2/data/11-785-f23-hw4p2/train-clean-100/transcripts"# path to the transcripts in the train clean 100 partition
            mfcc_files          = sorted(os.listdir(mfcc_dir))# create a list of paths for all the mfccs in the mfcc directory
            transcript_files    = sorted(os.listdir(transcript_dir))# create a list of paths for all the transcripts in the transcript directory

            mfcc_dir       = f"/home/yuwu3/4P2/data/11-785-f23-hw4p2/train-clean-360/mfcc"# path to the mfccs in the train clean 100 partition
            transcript_dir = f"/home/yuwu3/4P2/data/11-785-f23-hw4p2/train-clean-360/transcripts"# path to the transcripts in the train clean 100 partition
            mfcc_files += sorted(os.listdir(mfcc_dir))
            transcript_files += sorted(os.listdir(transcript_dir))

        assert len(mfcc_files) == len(transcript_files)
        length = len(mfcc_files)# TODO

        self.mfcc_dir = mfcc_dir
        self.transcript_dir = transcript_dir
        self.mfcc_files         = mfcc_files
        self.transcript_files   = transcript_files
        self.length             = len(transcript_files)
        print("Loaded file paths ME: ", partition)


    def __len__(self):
        return self.length

    def __getitem__(self, ind):

        # Load the mfcc and transcripts from the mfcc and transcript paths created earlier
        mfcc        =  np.load(os.path.join(self.mfcc_dir, self.mfcc_files[ind]))
        transcript  = np.load(os.path.join(self.transcript_dir, self.transcript_files[ind]))
         # NOT Remove [SOS] and [EOS] from the transcript

        # Normalize the mfccs and map the transcripts to integers
        mfcc                = cepstral_mean_normalization(mfcc)
        transcript_mapped   = transcript_to_ind(transcript)

        return torch.FloatTensor(mfcc), torch.LongTensor(transcript_mapped)

    def collate_fn(self,batch):
        # batch of input mfcc coefficients
        batch_mfcc, batch_transcript = zip(*batch) # TODO
        # Also be sure to check the input format (batch_first)
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True) # TODO
        lengths_mfcc = torch.tensor([len(seq) for seq in batch_mfcc])
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True) # TODO
        lengths_transcript = torch.tensor([len(seq) for seq in batch_transcript])
        return batch_mfcc_pad, batch_transcript_pad, lengths_mfcc, lengths_transcript

class SpeechDatasetTest(torch.utils.data.Dataset):

    def __init__(self, root, partition, cepstral=False):

        self.mfcc_dir   = f"/home/yuwu3/4P2/data/11-785-f23-hw4p2/test-clean/mfcc"# path to the test-clean mfccs
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        self.mfccs = []
        for i, filename in enumerate(self.mfcc_files):
            mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[i])) # load the mfccs
            if cepstral:
                # Normalize the mfccs
                mfcc = cepstral_mean_normalization(mfcc)# TODO
            # append the mfcc to the mfcc list created earlier
            self.mfccs.append(mfcc)
        self.length = len(self.mfccs)
        print("Loaded: ", partition)

    def __len__(self):
        return self.length # TODO

    def __getitem__(self, ind):
        mfcc = torch.FloatTensor(self.mfccs[ind])
        return mfcc

    # def collate_fn(self,batch):
    #     batch_x, lengths_x = [], []
    #     for x in batch:
    #         # Append the mfccs and their lengths to the lists created above
    #     # pack the mfccs using the pad_sequence function from pytorch
    #     batch_x_pad = # TODO
    #     return batch_x_pad, torch.tensor(lengths_x)
    
    def collate_fn(self,batch):
        batch_mfcc = batch # TODO
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True) # TODO
        lengths_mfcc = torch.tensor([len(seq) for seq in batch_mfcc])        
        return batch_mfcc_pad, lengths_mfcc