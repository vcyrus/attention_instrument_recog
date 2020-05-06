'''
   OpenMicDataset.py: src.datasets.OpenMicDataset

   Loads OpenMicDataset 10 second VGGish embeddings

   Cyrus Vahidi
'''

import os

import numpy as np
import pandas as pd

import torchaudio

from src.datasets.Dataset import Dataset

from sklearn.preprocessing import MinMaxScaler

class OpenMicDataset(Dataset):
    '''
        path: path to npz features file,
        partition_path: path to OpenMic partition file e.g split01_train.csv
    '''
    def __init__(self, path, partition_path=None, scaler=None):
        self.path = path
        self.partition_path = partition_path

        self.load_openmic()

        # scale feature in range [0, 1], reshaping to 2d first
        self.features2d = self.features.reshape(-1, self.features.shape[2])
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.features2d = self.scaler.fit_transform(self.features2d)
        else:
            # use the training set's scaler for the test data
            self.features2d = scaler.transform(self.features2d)
            

        # reshape to original
        (n_samples, n_timesteps, n_features) = self.features.shape
        self.features = self.features2d.reshape(n_samples, 
                                                n_timesteps, 
                                                n_features)

    def load_openmic(self):
        # this hack allows np.load to load the 'sample_key' attribute
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        openmic = np.load(os.path.join(self.path, 'openmic-2018.npz'))

        np.load = np_load_old

        self.sample_keys = openmic['sample_key']
        # load the partition csv
        df_part = pd.read_csv(self.partition_path, header=None)
        # get indexes of sample ids for the partition
        partition_idxs = np.where(np.isin(df_part[0].tolist(), self.sample_keys))

        self.features = openmic['X'][partition_idxs]
        self.observations = openmic['Y_mask'][partition_idxs]
        gt_probs = openmic['Y_true'][partition_idxs]
        self.labels = np.where(gt_probs > 0.5, 1, 0)

    def __getitem__(self, idx):
        sample = {
            "features": self.features[idx],
            "labels": self.labels[idx],
            "observations": self.observations[idx]
        }
        return sample

    def __len__(self):
        return len(self.features)
