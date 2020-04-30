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
                temporal_dim=10,
                dropout_rate=0.6):
        super(AttentionModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)

        self.attn = AttentionLayer(embedding_dim, n_classes)
        self.score = nn.Linear(embedding_dim, n_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(temporal_dim)
        self.bn2 = nn.BatchNorm1d(temporal_dim)
        self.bn3 = nn.BatchNorm1d(temporal_dim)


        self.n_classes = n_classes

    def forward(self, batch):
        z = self.dropout(self.bn1(F.relu(self.fc1(batch))))
        z = self.dropout(self.bn2(F.relu(self.fc2(z))))
        z = self.dropout(self.bn3(F.relu(self.fc3(z))))

        embedding = torch.add(batch, z)

        attn_weights = self.attn(embedding)
        score = self.score(embedding)

        score = torch.sum(torch.mul(attn_weights, score), dim=1)

        return score.view((-1, self.n_classes))

class AttentionLayer(nn.Module):
    def __init__(self, n_dims_in, n_dims_out):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(n_dims_in, n_dims_out)
    
    def forward(self, z):
        weights = torch.matmul(z, self.attn.weight.t())
        
        return F.softmax(weights, dim=2)
