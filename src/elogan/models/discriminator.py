import torch
from torch import nn

from ..constants import BOARD_SIZE


class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(1 + BOARD_SIZE * 2, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1),
        nn.Sigmoid(),
    )
  
  def forward(self, x):
    out = self.model(x)
    return out


discriminator = Discriminator()
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-7)
discriminator_loss_func = nn.BCELoss()
