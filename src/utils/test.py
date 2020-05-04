import torch
import torch.nn as nn

import numpy as np

from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader

def evaluate(model, device, dataset, label_to_int):
    print("Evaluating model ... ")
    model.eval()

    int_to_label = {v: k for k, v in label_to_int.items()}

    batch_size = 1
    generator = DataLoader(dataset, batch_size=batch_size)

    y_true = np.zeros((len(generator), len(label_to_int)))
    y_pred =np.zeros((len(generator), len(label_to_int)))

    for i, batch in enumerate(generator, 0):
        model.zero_grad()

        X, y = batch["features"], batch["labels"]
        X = X.to(device=device).float()
        y = y.numpy()

        outputs = torch.sigmoid(model(X))

        # threshold
        outputs = (outputs >= 0.5).float() * 1
        outputs = outputs.cpu().numpy()

        y_true[i] = y[0]
        y_pred[i] = outputs[0]
    
    import pdb; pdb.set_trace()
            
    print("Accuracy: ", accuracy_score(y_true, y_pred))
