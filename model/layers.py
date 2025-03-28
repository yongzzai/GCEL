'''
@author: Y.J.Lee
'''

import torch.nn as nn
from torch_geometric.nn import TransformerConv, PositionalEncoding, MLP

class FirstViewPreLayer(nn.Module):
    def __init__(self, node_dim:int = None, edge_dim:int = None, hidden_dim:int = 64):
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

        self.PosEnc = PositionalEncoding(out_channels=hidden_dim)
        self.AttrEmbedder = nn.ModuleList([nn.Embedding(1000, hidden_dim) for _ in range(self.num_attr)])   # Shape (num_events, hidden_dim*num_attr)

        self.EdgeTransform = nn.Sequential(
                             nn.Linear(int(hidden_dim*self.num_attr),hidden_dim),
                             nn.LayerNorm(hidden_dim))         # Shape (num_edges, hidden_dim)
    
    def forward(self,):
        pass    #TODO


class SecondViewPreLayer(nn.Module):
    def __init__(self, node_dim:int = None, edge_dim:int = None, hidden_dim:int = 64):
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

        self.PosEnc = PositionalEncoding(out_channels=hidden_dim)
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


class GraphEncoder(nn.Module):
    
    def __init__(self, hidden_dim:int = 64, num_layers:int = 1, dropout:float = 0.3):
        super(GraphEncoder, self).__init__()
        '''
        This layer is for the encoding of the graph.
        Structure of the encoder:
        TransformerConv -> Linear -> LayerNorm -> GELU -> Dropout

        :param hidden_dim: The dimension of the hidden.
        :param latent_dim: The dimension of the latent.
        :param num_layers: The number of layers.
        :param dropout: The dropout rate.
        '''

        self.Convs, self.DOs, self.LNs, = self._SetLayers(hidden_dim, num_layers, dropout)

        self.norm = nn.LayerNorm(hidden_dim) 
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = MLP([hidden_dim, hidden_dim*2, hidden_dim/2], norm=None)
    
    def _SetLayers(self, hidden_dim, num_layers, dropout):

        Convs, Lins, LNs, Acts, DOs  = [nn.ModuleList() for _ in range(5)]

        for i in range(num_layers):
            Convs.append(TransformerConv(in_channels=hidden_dim,
                                         out_channels=hidden_dim,
                                         heads=4,
                                         edge_dim=hidden_dim,
                                         beta=True,
                                         dropout=dropout,))
            Lins.append(nn.Linear(hidden_dim*4, hidden_dim))
            LNs.append(nn.LayerNorm(hidden_dim))
            Acts.append(nn.GELU())
            DOs.append(nn.Dropout(dropout))

        return Convs, Lins, LNs, Acts, DOs
    
    def forward(self):
        pass