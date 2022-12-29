import torch
import numpy as np

#Initialize a Tensor

#Raw data(array) => Tensor
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

#Numpy array => Tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#Tensor size => Tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

#Random or Constant Tensor
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#Attributes of a Tensor
#Shape, Datatype, Stored Device
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#Operations on Tensors
#Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
#Move Tensor from CPU to GPU
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    
#From now on, m1 chips can GPU accelerate
#https://bio-info.tistory.com/184

#Detail Tensor operations
#https://pytorch.org/docs/stable/torch.html

#Tensor indexing, slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

#Concat Tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
t2 = torch.cat([tensor, tensor, tensor], dim=0)
print(t2)

#Arithmetic operations
#matrix multiplication
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

#matrix element-wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#Single-element tensors
#item() convert single-element tensor to float type
agg = tensor.sum()
agg_item = agg.item()
print(agg, type(agg))
print(agg_item, type(agg_item))

#In-place operation store change value, matrix operation broadcasting
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

#Bridges with Numpy
#Tensors on the CPU and Numpy array share their memory location

#Tensor => numpy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

#memory location sharing
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

#Numpy array => Tensor
n = np.ones(5)
t = torch.from_numpy(n)
#memory sharing
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

