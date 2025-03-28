import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from typing import Optional

class FirstViewPreLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(FirstViewPreLayer, self).__init__()
        '''
        This layer is for the preprocessing of the first view graph.
        The first view is a graph with the following properties:
        1. Nodes: representing activities
        2. Edges: representing the relationships between activities
        3. Edge features: representing the attributes of the events i.e.,event ordering, event attributes

        :parm node_dim: The initial dimension of the node features.
        :param edge_dim: The initial dimension of the edge features.
        :param hidden_dim: The dimension of the hidden.
        '''
        
        # For nodes (activity)
        self.NodeTransform = nn.Sequential(
                             nn.Linear(node_dim, hidden_dim),
                             nn.LayerNorm(hidden_dim))       # Shape (num_nodes, hidden_dim)

        # For edges (events)
        self.num_attr = edge_dim - 1    # first dimension is the event ordering

        self.PosEnc = gnn.PositionalEncoding(out_channels=hidden_dim)
        self.AttrEmbedder = nn.ModuleList([nn.Embedding(1000, hidden_dim) for _ in range(self.num_attr)])   # Shape (num_events, hidden_dim*num_attr)

        self.EdgeTransform = nn.Sequential(
                             nn.Linear(int(hidden_dim*self.num_attr),hidden_dim),
                             nn.LayerNorm(hidden_dim))         # Shape (num_edges, hidden_dim)
    
    def forward(self,):
        pass    


class SecondViewPreLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(SecondViewPreLayer, self).__init__()
        '''
        This layer is for the preprocessing of the second view graph.
        The first view is a graph with the following properties:
        1. Nodes: representing events
        2. Node features: representing the attributes of the events i.e.,event ordering, event attributes
        3. Edges: representing the activities

        :param node_dim: The initial dimension of the node features.
        :param edge_dim: The initial dimension of the edge features.
        :param hidden_dim: The dimension of the hidden.
        '''

        # For nodes (events)
        self.num_attr = node_dim - 1   # first dimension is the event ordering

        self.PosEnc = gnn.PositionalEncoding(out_channels=hidden_dim)
        self.AttrEmbedder = nn.ModuleList([nn.Embedding(1000, hidden_dim) for _ in range(self.num_attr)])

        self.NodeTransform = nn.Sequential(
                             nn.Linear(int(hidden_dim*self.num_attr), hidden_dim),
                             nn.LayerNorm(hidden_dim))          # Shape (num_nodes, hidden_dim)
        
        # For edges (activities)
        self.EdgeTransform = nn.Sequential(
                             nn.Linear(edge_dim, hidden_dim),
                             nn.LayerNorm(hidden_dim))          # Shape (num_edges, hidden_dim)

    def forward(self, x):
        pass


