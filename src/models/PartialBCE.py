import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialBCE(nn.Module):

    def __init__(self):
        super(PartialBCE, self).__init__()

    def forward(self, y_pred, y_true, y_obs):
        obs_weight = y_obs.float()

        return F.binary_cross_entropy_with_logits(y_pred, 
                                                  y_true, 
                                                  weight=obs_weight, 
                                                  reduction='mean')
