import numpy as np

def sigmoid(x):
    x = 1/(1+np.exp(x))
    return x
t = np.array([[1],[1],[2],[1],[-1]])
a = np.array([[1,1,1,1],[1,0,0,0],[1,0,0,0],[1,1,1,1]])
c = np.zeros((5,4))
r = np.array([[0.5],[1.2],[0.3],[0.7],[-1],[9]])
print(sigmoid(np.dot(c,t[0:-1])+t[-1]).shape)
print(r.round().shape)



e = np.zeros_like(t)

e[0:-1] = t[0:-1]
print(e)
print(e[0:-1].shape, t[0:-1].shape)

print(np.random.randn(1,2))