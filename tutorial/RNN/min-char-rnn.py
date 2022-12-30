import numpy as np

data = open("yejin.txt",'r').read()
chars = list(set(data)) # Unique vocab of data
data_size, vocab_size = len(data), len(chars)

print(f"data has {data_size} characters, {vocab_size} unique.")

char_to_idx = {c:i for i,c in enumerate(chars)} # char -> idx embedding
idx_to_char = {i:c for i,c in enumerate(chars)} # To return idx -> char

#hyperparameters
hidden_size = 500
seq_length = 25 #nums of unfolded rnn length each epoch (length of sentence)
learning_rate = 1e-3

#model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01
Whh = np.random.randn(hidden_size, hidden_size)*0.01
Why = np.random.randn(vocab_size, hidden_size)*0.01
#hidden bias
bh = np.zeros((hidden_size, 1))
#ouput bias
by = np.zeros((vocab_size, 1))

def loss_fn(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev) #hs[-1] is zero vector
    loss = 0
    
    #forward
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1)) 
        #xs -> dict: {t : [vocab_size x 1]} "Warning! xs[t] is 2D array do not confuse!""
        xs[t][inputs[t]] = 1 
        #xs: dict t : [vocab_size x 1] <- vocab idx = 1<- inputs[t]: vocab idx of char
        hs[t] = np.tanh(np.dot(Wxh, xs[t])+np.dot(Whh, hs[t-1])+bh) 
        #hidden state, hs -> dict: {t : [hidden_size x 1]} <- hidden feature 
        ys[t] = np.dot(Why, hs[t])+by 
        # feature output vector of each rnn, ys -> dict: {t : [vocab_size x 1]} <- unnormalized prob 
        ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t])) 
        #ps -> dict: {t : [vocab_size x 1] } <- normalized prob by softmax
        loss += -np.log(ps[t][targets[t],0]) 
        #ps t-th list[vocab_size x 1] and targets[t]-th value -> [value] type, so add 0(numpy array option cannot work in python list)
    
    #backward
    dWxh,  dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhprev = np.zeros_like(hs[0])
    
    for t in reversed(range(len(inputs))): #calculate n -> 1 reverse
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # dy = y-t
        dWhy += np.dot(dy, hs[t].T) # y = why @ h + by, dwhy = dy @ h.T
        dby += dy # dby = dy
        dh = np.dot(Why.T, dy) + dhprev # h = tanh(wxhX+whhH)
        dhraw = (1-hs[t]*hs[t])*dh # tanh
        dbh += dhraw # dbh = dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhprev = np.dot(Whh.T, dhraw) # give prev dh value
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam,-5,5,out=dparam) # prevent greident overflow
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1] #hs[len(inputs)-1] since calc was reverse, it means h0 value

def sample(h, seed_idx, n):
    x = np.zeros((vocab_size, 1))
    x[seed_idx] = 1
    idxes = []
    
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x)+np.dot(Whh, h)+bh)
        y = np.dot(Why, h)+by
        p = np.exp(y)/np.sum(np.exp(y))
        idx = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        idxes.append(idx)
    return idxes


def run(epoch:int) -> None:
    n, p = 0, 0
    epochs = epoch+1
    
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by)
    smooth_loss = -np.log(1.0/vocab_size)*seq_length
    
    while n < epochs:
        if p+seq_length+1 >= len(data) or n==0:
            hprev = np.zeros((hidden_size, 1))
            p = 0
        inputs = [char_to_idx[c] for c in data[p:p+seq_length]]
        targets = [char_to_idx[c] for c in data[p+1:p+seq_length+1]]
        
        if n%100 == 0:
            sample_idx = sample(hprev, inputs[0], 30)
            txt = ''.join(idx_to_char[idx] for idx in sample_idx)
            print(f"----\n {txt} \n----")
            
        
        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_fn(inputs, targets, hprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n%100 == 0: print(f"iter {n}, loss: {smooth_loss}")
        
        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                    [dWxh, dWhh, dWhy, dbh, dby], 
                                    [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
        p += seq_length # move data pointer
        n += 1 # iteration counter
        
run(2000)