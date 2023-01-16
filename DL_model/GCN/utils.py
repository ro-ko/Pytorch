import numpy as np
import scipy.sparse as sp
import torch

'''
func make sequence
load_data -> encode_onehot -> normalize -> sparse_mx_to_torch_sparse_tensor / accuracy
'''

def encode_onehot(labels):
    classes = set(labels)
    # class table class : on-hot code
    classes_dict = {c: np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype=np.int32)
    
    return labels_onehot

def load_data(path="../../Data/cora/",dataset="cora"):
    print(f'Loading {dataset} dataset')
    
    # read txt from cora as string
    # content file : <paper_id>123 <word_attributes>(0100 ~ 1010)+ <class_label>(neural network)
    idx_feature_labels = np.genfromtxt(f"{path}{dataset}.content",dtype=np.dtype(str))
    #print(idx_feature_labels.shape) -> (2708, 1435)
    
    features = sp.csr_matrix(idx_feature_labels[:,1:-1], dtype=np.float32)
    # make class label to one hoot code e.g. {100000 : neural netowork}
    labels = encode_onehot(idx_feature_labels[:,-1])
    
    #build graph
    # paper_id list
    idx = np.array(idx_feature_labels[:,0], dtype=np.int32)
    # dict : paper_id : index
    idx_map = {j: i for i, j in enumerate(idx)}
    
    # read edge as list
    edges_unordered = np.genfromtxt(f"{path}{dataset}.cites",dtype=np.int32)
    #print(edges_unordered.shape) -> (5429, 2)
    
    # after flatten convert paper_id to index of idx reshape edge shape
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    
    # coo_matrix -> edge_num, node1, node2, shape=node_num * node_num
    adj = sp.coo_matrix((np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),shape=(labels.shape[0],labels.shape[0]),dtype=np.float32)
    #print(adj.shape) -> (2708, 2708)

    # build symmetric adjacency matrix
    # suppose that adj is undirected graph, cotains duplicate edges(edge weight)
    # sp.coo_matrix.multiply -> elementwise product 
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    
    # make sparse to dense
    features = torch.FloatTensor(np.array(features.todense()))
    # length one-hot code
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    idx_val = torch.LongTensor(idx_val)
    
    return adj, features, labels, idx_train, idx_val, idx_test

# mx -> matrix
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # rowsum inversion
    r_inv = np.power(rowsum, -1).flatten()
    # make inf to 0.
    r_inv[np.isinf(r_inv)] = 0.
    # make diag mx
    r_mat_inv = sp.diags(r_inv)
    # normalize
    ma = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sprase mx to a torch sparse tensor"""
    # spare mx -> coo matrix
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # vertical stack row, col
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # value of mx
    values = torch.from_numpy(sparse_mx.data)
    
    shape = torch.Size(sparse_mx.shape)
    
    # convert scipy sparse to pytorch sparse
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    
    preds = output.max(1)[1].type_as(labels)
    # return double type tensor
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)