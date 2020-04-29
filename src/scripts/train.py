import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam  

from src.models.AttentionModel import AttentionModel

from src.utils.training import train, get_datasets
from src.utils.parse_args import parse_args
from src.utils.test import evaluate

if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(os.path.join(args.openmic_path, 
                                  'openmic-2018-aggregated-labels.csv'))
    label_to_idx = {label: idx 
              for idx, label in enumerate(np.sort(df['instrument'].unique()))}

    
    datasets = get_datasets(
                  args.openmic_path, 
                  args.val_split,
                  args.test_split
              )
    n_classes = len(label_to_idx)

    writer = SummaryWriter()

    cuda_device = torch.device('cuda' if args.use_cuda and \
                                      torch.cuda.is_available() \
                                      else 'cpu')
    print('Device: {}'.format(cuda_device))

    model = AttentionModel()

    optimizer = Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')

    model, criterion = train(
        datasets, 
        args.batch_size,
        args.n_epochs,
        args.lr,
        cuda_device,
        model,
        optimizer,
        writer,
        args.weight_decay, 
        args.audio_len,
        criterion,
        args.patience,
    )
    writer.close()

    # Save the model
    if args.out_path:
        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)
        path = os.path.join(args.out_path, 
                          "model_{0}".format(args.transform))
        torch.save(model.state_dict(), path)

    confusions = evaluate(model, 
                          cuda_device, 
                          datasets['test'], 
                          label_to_idx, 
                          criterion)
