import torch
from torch import nn

from ..constants import BOARD_SIZE, k


class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(BOARD_SIZE, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1),
    )
  
  def forward(self, x):
    out = self.model(x)
    return out


def gen_loss(y_true, y_pred):
  loss = torch.max(y_pred,0)[0] - y_pred * y_true
  loss += k * torch.log(1+torch.exp((-1)*torch.abs(y_pred)))
  return torch.mean(loss)


generator = Generator()
generator_optim = torch.optim.Adam(generator.parameters(), lr=1e-4)
generator_loss_func = gen_loss
