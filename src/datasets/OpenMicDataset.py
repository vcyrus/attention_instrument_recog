'''
   OpenMicDataset.py: src.datasets.OpenMicDataset

   Loads OpenMicDataset 10 second VGGish embeddings

   Cyrus Vahidi
'''

import os

import torchaudio

from src.datasets.Dataset import Dataset

class OpenMicDataset(Dataset):
    def __init__(self, 
                path):
        self.path = path

    def walk_path(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)
