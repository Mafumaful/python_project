import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt

class self_designed_MessagePassingLayer(MessagePassing):
    def __init__(self, aggr='max'):
        super(self_designed_MessagePassingLayer, self).__init__(aggr=aggr)
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        return 0.5 * x_i + 2 * x_j

    def update(self, aggr_out , x):
        return x + 0.5 * aggr_out
    
x = torch.tensor([[6, 4],
                  [0, 1],
                  [5, 3],
                  [1, 2]])

edge_index = torch.tensor(
    [[0, 1, 0, 2, 1, 2, 2, 3],
     [1, 0, 2, 0, 2, 1, 3, 2]]
)

edge_attr = torch.tensor([[1],
                          [1],
                          [4],
                          [4],
                          [2],
                          [2],
                          [5],
                          [5]])

y = torch.tensor([
    [1],
    [0],
    [1],
    [0]
])

graph = Data(x = x, edge_index= edge_index, edge_attr=edge_attr, y=y)
print(graph)

graph = Data(x = x, edge_index= edge_index, edge_attr=edge_attr, y=y)

self_layer = self_designed_MessagePassingLayer(aggr='max')
print(f"After 1 mp layer, graph.x = {self_layer(graph.x, graph.edge_index)}")