import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from utils import load_data, accuracy
from model import GCN
import matplotlib.pyplot as plt

# training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()



# Model and Optimizer
model = GCN(nfeat=features.shape[1],
            nhid = args.hidden,
            nclass = labels.max().item()+1,
            dropout = args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr = args.lr,
                       weight_decay = args.weight_decay)

if args.cuda:
    model.cuda()
    features= features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
def train(epoch):
    t = time.time()
    # start train
    model.train()
    # reset grad
    optimizer.zero_grad()
    # forward
    output = model(features, adj)
    # clac loss
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # backpropagation
    loss_train.backward()
    # update parameters
    optimizer.step()
    
    if not args.fastmode:
        model.eval()
        output = model(features, adj)
        
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if epoch % 10 ==0:
        print(f'Epoch: {epoch+1:04d}',f'loss_train: {loss_train.item():.4f}',f'acc_train: {acc_train.item():.4f}',f'loss_val: {loss_val:.4f}', f'acc_val: {acc_val:.4f}', f'time: {time.time()-t:.4f}')
    return loss_train.item(), loss_val.item()
    
def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",f"loss= {loss_test.item():.4f}",f"accuracy= {acc_test.item():.4f}")
    
# Train model
t_total = time.time()

train_loss = []
val_loss = []

for epoch in tqdm(range(args.epochs)):
    tloss, vloss = train(epoch)
    train_loss.append(tloss)
    val_loss.append(vloss)
print("Train end!")
print(f"Total time elased: {time.time()-t_total}")

fig = plt.figure()

plt.plot(train_loss, 'b', label='Train')
plt.plot(val_loss, 'r', label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.savefig("loss.png")

# Test
test()