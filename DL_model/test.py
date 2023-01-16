import numpy as np
import scipy.sparse as sp
import torch
def sigmoid():
    x = 10
    x = 1/(1+np.exp(x))
    return x
t = np.array([[1],[1],[2],[1],[-1]])
a = np.array([[1,1,1,1],[1,0,0,0],[1,0,0,0],[1,1,1,1]])

x = sigmoid()
print(x)

import matplotlib.pyplot as plt

b = np.random.random_sample(size=100)
fig = plt.figure()

plt.plot(b, 'b', label='Train')
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.show()