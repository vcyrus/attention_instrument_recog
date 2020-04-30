import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam  

from src.models.AttentionModel import AttentionModel

from src.utils.training import train, get_datasets_partitioned
from src.utils.parse_args import parse_args
from src.utils.test import evaluate

if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(os.path.join(args.openmic_path, 
                                  'openmic-2018-aggregated-labels.csv'))
    label_to_idx = {label: idx 
              for idx, label in enumerate(np.sort(df['instrument'].unique()))}
   
    train_part_path = os.path.join(os.path.join(args.openmic_path, "partitions"),
                                  "split01_train.csv")
    test_part_path = os.path.join(os.path.join(args.openmic_path, "partitions"), 
                                  "split01_test.csv")
    datasets = get_datasets_partitioned(args.openmic_path, 
                                        args.val_split,
                                        train_part_path,
                                        test_part_path)
    n_classes = len(label_to_idx)

    writer = SummaryWriter()

    cuda_device = torch.device('cuda' if args.use_cuda and \
                                      torch.cuda.is_available() \
                                      else 'cpu')
    print('Device: {}'.format(cuda_device))

    model = AttentionModel(n_classes, 
                          embedding_dim=args.embedding_dim,
                          dropout_rate=args.dropout_rate)

    optimizer = Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    model = train(
        datasets, 
        args.batch_size,
        args.n_epochs,
        args.lr,
        cuda_device,
        model,
        optimizer,
        writer,
        args.weight_decay, 
        criterion,
        args.patience,
    )
    writer.close()

    evaluate(model, device, dataset, label_to_int)

    # Save the model
    if args.out_path:
        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)
        path = os.path.join(args.out_path, "model")
        torch.save(model.state_dict(), path)
