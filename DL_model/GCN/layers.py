import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules import Module

class GraphConvolution(Module):
    """_summary_
    Simple GCN layer
    Args:
    """
    
    def __init__(self, in_features, out_features, bias=True) -> None:
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            # adds a parameter to the module, None -> cuda ignore 'bias'
            self.register_parameter('bias', None)
        self.reset_parameter()
    
    # Xavier intialization for tanh?
    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        return self.__class__.__name__+'('+str(self.in_features)+'->'+str(self.out_features)+')'