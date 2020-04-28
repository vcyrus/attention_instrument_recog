'''
   OpenMicDataset.py: src.datasets.OpenMicDataset

   Loads OpenMicDataset 10 second VGGish embeddings

   Cyrus Vahidi
'''

import os

import numpy as np

import torchaudio

from src.datasets.Dataset import Dataset

class OpenMicDataset(Dataset):
    '''
        path: path to npz file
    '''
    def __init__(self, path):
        self.path = path

        self.load_openmic()

    def load_openmic(self):
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        openmic = np.load(os.path.join(self.path, 'openmic-2018.npz'))

        np.load = np_load_old

        self.features = openmic['X']
        self.labels = openmic['Y_true']
        self.sample_keys = openmic['sample_key']

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.features)
