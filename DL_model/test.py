import torchvision.models.vgg as vgg
import torch.nn as nn
import torch
from typing import Any, cast, Dict, List, Union
import VGG



test = VGG.vgg11(num_classes=1000,dropout=0.5)
print(test)