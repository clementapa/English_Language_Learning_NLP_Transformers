import torch.nn as nn
import torch

class MCRMSELoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, outputs, targets):
        """https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/348985"""
        assert outputs.shape == targets.shape

        colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        return loss
