import numpy as np
import matplotlib.pyplot as plt

# data num
N = 100
# dimension
D = 2
# label num
K = 3

X = np.zeros((N*K,D)) # N*K x D array
y = np.zeros(N*K, dtype='uint8')



# polar coordinate not cartesian
for i in range(K):
    ix = range(N*i, N*(i+1))
    r = np.linspace(0.0, 1, N) # Generate N nums between 0.0 ~ 1.0 sequentially, radius, N x 1 array 
    t = np.linspace(i*4, (i+1)*4, N) + np.random.randn(N)*0.2 # theta, N x 1 array
    X[ix] = np.c_[r*np.sin(t),r*np.cos(t)] # r,t are 1-dim array -> concatenate into column
    y[ix] = i #label
    
plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral)
plt.show()


#Regression
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

step_size = 1e-1
reg = 1e-3

num_examples = X.shape[0]

for i in range(500):
    #forward
    scores = np.dot(X,W)+b #
    
    #loss
    exp_scores = np.exp(scores-np.max(scores))
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    # For multi-class classification, need to keep dims
    
    log_probs = -np.log(probs[range(num_examples),y]) # Need study about numpy.array skills -> ???????????
    data_loss = np.sum(log_probs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W) # 0.5 is constant to derivative
    
    loss = data_loss + reg_loss
    if i%10==0:
        print(f"iteration {i}: loss {loss}")
    
    #backward
    dy = probs
    dy[range(num_examples),y] -= 1
    dy /= num_examples
    
    dW = np.dot(X.T, dy)
    dW += reg*W
    
    db = np.sum(dy,axis=0,keepdims=True)
    
    W -= step_size*dW
    b -= step_size*db
    
scores = np.dot(X,W) + b
predict = np.argmax(scores,axis=1)

print(f"train_acc : {np.mean(predict == y):.2f}")

plt.scatter(X[:,0],X[:,1],c=predict,s=40,cmap=plt.cm.Spectral)
plt.show()


#################################################################################
#################################################################################
#################################################################################


#2-dims NN
h = 100

W1 = 0.01 * np.random.randn(D,h) 
b1 = np.zeros((1,h))

W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))

activation = "ReLU"
step_size = 1e-0
reg = 1e-3

num_examples = X.shape[0]

for i in range(10000):
    #forward
    h1 = np.dot(X,W1) + b1
    h1 = np.maximum(0, h1)
    scores = np.dot(h1,W2) + b2
    
    #loss
    exp_scores = np.exp(scores-np.max(scores)) # more stable than np.exp(y)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    
    log_probs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(log_probs)/num_examples
    reg_loss = 0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2)) # 0.5 is constant to derivative
    
    loss = data_loss + reg_loss
    
    if i%1000==0:
        print(f"iteration {i}: loss {loss}")
    
    #backward
    dy = probs
    dy[range(num_examples),y] -= 1
    dy /= num_examples
    
    dW2 = np.dot(h1.T, dy)
    dW2 += reg*W2
    db2 = np.sum(dy,axis=0,keepdims=True)
    
    dh1 = np.dot(dy, W2.T)
    dh1[h1<=0] = 0
    
    dW1 = np.dot(X.T, dh1)
    dW1 += reg*W1
    db1 = np.sum(dh1,axis=0,keepdims=True)
    
    W2 -= step_size*dW2
    b2 -= step_size*db2
    W1 -= step_size*dW1
    b1 -= step_size*db1
    

h1 = np.dot(X,W1) + b1
h1 = np.maximum(0, h1)
scores = np.dot(h1,W2) + b2

predict = np.argmax(scores,axis=1)

print(f"train_acc : {np.mean(predict == y):.2f}")

plt.scatter(X[:,0],X[:,1],c=predict,s=40,cmap=plt.cm.Spectral)
plt.show()
