import torch
import torch.nn as nn
from typing import Any, Dict, List, Union

__all__ = [
    "VGG",
    "VGG11"
    "VGG11_BN"
    "VGG13"
    "VGG13_BN"
    "VGG_16"
    "VGG19"
    "VGG19_BN"
]
#function -> make _layers -> _vgg -> VGG
class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x)
        x = self.classifier(x)
        return x

# M : maxpooling, int : filter num
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v=="M":
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str,int]]]={ 
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

# **kwargs : num_classes, dropout
def _vgg(cfg: str, batch_norm:bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg11(**kwargs):
    return _vgg("A", False, **kwargs)

def vgg11_bn(**kwargs):
    return _vgg("A", True, **kwargs)

def vgg13(**kwargs):
    return _vgg("B", False, **kwargs)

def vgg13_bn(**kwargs):
    return _vgg("B", True, **kwargs)

def vgg16(**kwargs):
    return _vgg("C", False, **kwargs)

def vgg16_bn(**kwargs):
    return _vgg("C", True, **kwargs)

def vgg19(**kwargs):
    return _vgg("D", False, **kwargs)

def vgg19_bn(**kwargs):
    return _vgg("D", True, **kwargs)