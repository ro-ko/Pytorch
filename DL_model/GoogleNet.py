import torch
from torch.nn import nn

class GoogleNet(nn.module):
    def __init__(self) -> None:
        super().__init__()