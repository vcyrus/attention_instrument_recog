import torch
import torch.nn as nn

import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

from torch.utils.data import DataLoader

def evaluate(model, device, dataset, label_to_int):
    print("Evaluating model ... ")
    model.eval()

    int_to_label = {v: k for k, v in label_to_int.item()}

    batch_size = 32
    generator = DataLoader(dataset, batch_size=batch_size)

    for i, batch in enumerate(generator, 0):
        model.zero_grad()

        X, y = batch["features"], batch["label"]
        X = X.to(device=cuda_device).float()
        y = y.numpy()

        outputs = torch.sigmoid(model(X), dim=1)
        outputs = torch.where(outputs >= 0.5, 1, 0)
        _, preds = torch.max(outputs, 1)

