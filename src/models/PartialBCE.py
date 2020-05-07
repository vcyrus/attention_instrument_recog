import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialBCE(nn.Module):
    ''' 
        Computes a weighted cross entropy, given a matrix of observations
        for each label in each batch, where an observation is where there
        exists a known pos/neg observation for the class
        Hyperparameters alpha, beta, gamma from the paper
    '''

    def __init__(self, device=None):
        super(PartialBCE, self).__init__()
        self.alpha = 1
        self.beta = 1 
        self.gamma = -1
        self.device = device

    def forward(self, y_pred, y_true, y_obs):
        obs_weight = y_obs.float()

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        loss = loss * obs_weight

        n_obs = torch.sum(y_obs, dim=1)
        n_labels = y_obs.shape[1]
        p_obs = n_obs / torch.full((y_obs.shape[0], ), n_labels).to(self.device)
        p_obs = torch.pow(p_obs, self.gamma) / n_labels
        return torch.mean(p_obs * torch.sum(loss, dim=1))
