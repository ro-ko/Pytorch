import os
import networkx as nx
import pandas as pd

edgelist = pd.read_csv("../../Data/cora/cora.cites", sep='\t', header=None, names=["target", "source"])
edgelist['label'] = "cites"

Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
nx.set_node_attributes(Gnx,"paper",'label')

