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
                dropout_rate=0.6,
                attention=True):
        super(AttentionModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        
        self.attention = attention
        if attention:
            self.attn = AttentionLayer(embedding_dim, n_classes)
        self.score = nn.Linear(embedding_dim, n_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(temporal_dim)
        self.bn2 = nn.BatchNorm1d(temporal_dim)
        self.bn3 = nn.BatchNorm1d(temporal_dim)

        self.n_classes = n_classes

    def forward(self, batch):
        z1 = self.dropout(F.relu(self.bn1(self.fc1(batch))))
        z2 = self.dropout(F.relu(self.bn2(self.fc2(z1))))
        z3 = self.fc3(z2)
        embedding = self.dropout(F.relu(self.bn3(z3)))
        residual = torch.add(batch, embedding)
        
        score = self.score(residual)
        if self.attention:
            attn_weights = self.attn(residual)
            score = torch.sum(torch.mul(attn_weights, score), dim=1)
        else:
            score = torch.mean(self.score(score), dim=1)

        return score.view((-1, self.n_classes))

class AttentionLayer(nn.Module):
    def __init__(self, n_dims_in, n_dims_out):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(n_dims_in, n_dims_out)
    
    def forward(self, z):
        weights = torch.matmul(z, self.attn.weight.t())
        
        return F.softmax(weights, dim=2)
