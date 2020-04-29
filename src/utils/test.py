import torch
import torch.nn as nn

import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

from torch.utils.data import DataLoader

def evaluate(model, device, dataset, label_to_int, criterion):
    raise NotImplementedError
