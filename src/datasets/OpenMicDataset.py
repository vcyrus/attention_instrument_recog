'''
   OpenMicDataset.py: src.datasets.OpenMicDataset

   Loads OpenMicDataset 10 second VGGish embeddings

   Cyrus Vahidi
'''

import os

import numpy as np

import torchaudio

from src.datasets.Dataset import Dataset

from sklearn.preprocessing import MinMaxScaler

class OpenMicDataset(Dataset):
    '''
        path: path to npz file
    '''
    def __init__(self, path, scaler=None):
        self.path = path

        self.load_openmic()

        # scale feature in range [0, 1], reshaping to 2d first
        self.features2d = self.features.reshape(-1, self.features.shape[2])
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.features2d = self.scaler.fit_transform(self.features2d)
        else:
            self.features2d = scaler.transform(self.features2d)

        # reshape to original
        (n_samples, n_timesteps, n_features) = self.features.shape
        self.features = self.features2d.reshape(n_samples, 
                                                n_timesteps, 
                                                n_features)

    def load_openmic(self):
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        openmic = np.load(os.path.join(self.path, 'openmic-2018.npz'))

        np.load = np_load_old

        self.features = openmic['X']
        self.labels = openmic['Y_mask']
        self.sample_keys = openmic['sample_key']

    def __getitem__(self, idx):
        sample = {
            "features": self.features[idx],
            "labels": self.labels[idx]
        }
        return sample

    def __len__(self):
        return len(self.features)
