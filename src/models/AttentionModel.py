import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from src.models.Model import Model

class AttentionModel(Model):
    def __init__(self, 
                n_classes=20, 
                input_dim=128, 
                embedding_dim=128, 
                temporal_dim=10
                dropout_rate=0.6):
        super(AttentionModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)

        self.attn = AttentionLayer(temporal_dim, n_classes)
        self.pred = nn.Linear(embedding_dim, n_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)

    def forward(self, batch):
        z = self.dropout(self.bn1(F.relu(self.fc1(batch))))
        z = self.dropout(self.bn2(F.relu(self.fc2(z))))
        z = self.dropout(self.bn3(F.relu(self.fc3(z))))

        embedding = torch.add(batch, z)

        score = self.attn(torch.tranpose(embedding, 1, 2))
        pred = self.pred(embedding)

        y = nn.Sigmoid(torch.mul(score, pred))

        return y

def AttentionLayer(nn.Module):
    def __init__(self, n_dims_in, n_dims_out):
        super(AttentionModel, self).__init__()

        self.score = nn.Linear(n_dims_in, n_dims_out)
    
    def forward(self, batch):
        z = self.score(torch.transpose(embedding, 1, 2))
        
        return z
