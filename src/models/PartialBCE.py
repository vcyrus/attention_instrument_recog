import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialBCE(nn.Module):
    ''' 
        Computes a weighted cross entropy, given a matrix of observations
        for each label in each batch, where an observation is where there
        exists a known pos/neg observation for the class
        Hyperparameters alpha, beta, gamma from the paper

        Applies mask to loss values that have no observations (sets them to 0)
    '''

    def __init__(self, device=None):
        super(PartialBCE, self).__init__()
        self.alpha = 1
        self.beta = 1 
        self.gamma = -1
        self.device = device

    def forward(self, y_pred, y_true, y_mask):
        mask = y_mask.float()

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')

        # apply the observation mask to the loss
        loss = loss * mask

        # number of observations provided by the mask
        n_observation = torch.sum(mask, dim=1)

        # total number of labels
        n_labels = mask.shape[1]

        # proportion of labels that have observations
        p_obs = n_observation / torch.full((mask.shape[0], ), n_labels).to(self.device)

        # loss normalization term from the paper
        p_obs_normalizer = torch.pow(p_obs, self.gamma) / n_labels

        return torch.mean(p_obs_normalizer * torch.sum(loss, dim=1))
