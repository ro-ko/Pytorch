import platform
import torch

print (f"PyTorch version:{torch.__version__}")
print(f"MPS device build: {torch.backends.mps.is_built()}")
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")
print(platform.platform())