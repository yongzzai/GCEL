'''
@author: Y.J.Lee
'''

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, PositionalEncoding, MLP, Set2Set
from utils.featmask import maskNodes


class GraphEncoder(nn.Module):
    
    def __init__(self, node_dim:int = None, edge_dim:int = None, hidden_dim:int = 64, num_layers:int = 1, dropout:float = 0.3):
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

        self.FirstViewPreLayer = FirstViewPreLayer(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.SecondViewPreLayer = SecondViewPreLayer(node_dim=edge_dim, edge_dim=node_dim, hidden_dim=hidden_dim)

        self.Convs, self.Lins, self.LNs, self.Acts, self.DOs = self._SetLayers(hidden_dim, num_layers, dropout)

        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.pooling = Set2Set(hidden_dim, processing_steps=4)  # Shape (num_graphs, hidden_dim*2)

        self.mlp = MLP([hidden_dim*2, hidden_dim*4, int(hidden_dim/2)], norm=None)
    
    def _SetLayers(self, hidden_dim, num_layers, dropout):

        Convs, Lins, LNs, Acts, DOs  = [nn.ModuleList() for _ in range(5)]

        for i in range(num_layers):
            Convs.append(TransformerConv(in_channels=hidden_dim,
                                         out_channels=hidden_dim,
                                         heads=4,
                                         edge_dim=hidden_dim,
                                         beta=True,
                                         dropout=dropout))
            Lins.append(nn.Linear(hidden_dim*4, hidden_dim))
            LNs.append(nn.LayerNorm(hidden_dim))
            Acts.append(nn.GELU())
            DOs.append(nn.Dropout(dropout))

        return Convs, Lins, LNs, Acts, DOs
    
    def forward(self, data, train:bool=True):
        '''
        input: Shape(num_nodes, hidden_dim), Shape(num_edges, hidden_dim)

        _s: source graphs
        _t: transformed graphs
        '''
        x_s, edge_index_s, edge_attr_s, batch_s = data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch
        x_t, edge_index_t, edge_attr_t, batch_t = data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch

        if train:
            # First view
            # First view is anchor graph, so add further augmentation
            x1 = maskNodes(x=x_s, batch_idx=batch_s, p=0.3)
            x1, e1 = self.FirstViewPreLayer(x1, edge_attr_s)
            
            # Second view
            x2, e2 = self.SecondViewPreLayer(x_t, edge_attr_t)

            for idx in range(len(self.Convs)):
                
                x1_in = x1 if idx == 0 else h1
                
                h1 = self.Convs[idx](x1_in, edge_index_s, e1)
                h1 = self.Lins[idx](h1)
                h1 = self.LNs[idx](h1)
                h1 = self.Acts[idx](h1)
                h1 = self.DOs[idx](h1)

                # Residual connection
                h1 = h1 + x1_in

                x2_in = x2 if idx == 0 else h2

                h2 = self.Convs[idx](x2_in, edge_index_t, e2)
                h2 = self.Lins[idx](h2)
                h2 = self.LNs[idx](h2)
                h2 = self.Acts[idx](h2)
                h2 = self.DOs[idx](h2)

                # Residual connection
                h2 = h2 + x2_in
            
            z1, z2 = self.norm(h1), self.norm(h2)  # Shape (num_nodes, hidden_dim), (num_edges, hidden_dim)
            z1, z2 = self.linear(z1), self.linear(z2)  # Shape (num_nodes, hidden_dim), (num_edges, hidden_dim)
            z1, z2 = self.pooling(z1, batch_s), self.pooling(z2, batch_t) # Shape (num_graphs, hidden_dim*2)

        # For test
        else:
            # First view
            x1, e1 = self.FirstViewPreLayer(x_s, edge_attr_s)

            for idx in range(len(self.Convs)):
                
                x1_in = x1 if idx == 0 else h1
                
                h1 = self.Convs[idx](x1_in, edge_index_s, e1)   # Shape(num_nodes, hidden_dim*4)
                h1 = self.Lins[idx](h1)                         # Shape(num_nodes, hidden_dim)
                h1 = self.LNs[idx](h1)
                h1 = self.Acts[idx](h1)
                h1 = self.DOs[idx](h1)
                # Residual connection
                h1 = h1 + x1_in

            z1 = self.norm(h1)
            z1 = self.linear(z1)
            z1 = self.pooling(z1, batch_s)

            return z1

        # for computing loss
        dense_z1, dense_z2 = self.mlp(z1), self.mlp(z2)  # Shape (num_graphs, hidden_dim/2)

        return z1, z2, dense_z1, dense_z2




class FirstViewPreLayer(nn.Module):
    def __init__(self, node_dim:int = None, edge_dim:int = None, hidden_dim:int = 64):
        super(FirstViewPreLayer, self).__init__()
        '''
        This layer is for the preprocessing of the first view graph.
        The first view is a graph with the following properties:
        1. Nodes: representing activities
        2. Edges: representing the relationships between activities
        3. Edge features: representing the attributes of the events i.e.,event ordering, event attributes

        :param node_dim: The initial dimension of the node features.
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
        self.AttrEmbedder = nn.ModuleList([nn.Embedding(10000, hidden_dim) for _ in range(self.num_attr)])   # Shape (num_events, hidden_dim*num_attr)

        self.EdgeTransform = nn.Sequential(
                             nn.Linear(int(hidden_dim*self.num_attr) if self.num_attr > 0 else hidden_dim, hidden_dim),
                             nn.LayerNorm(hidden_dim))         # Shape (num_edges, hidden_dim)
    
    def forward(self, x_s, edge_attr_s):
        '''
        for the preprocessing of the first view graph.
        '''
        h_x = self.NodeTransform(x_s)  # Shape (num_nodes, hidden_dim)

        if self.num_attr>0:

            pos_vector = self.PosEnc(edge_attr_s[:,0]).repeat(1, self.num_attr)  # Shape (num_edges, hidden_dim*num_attr)
            
            embs = [self.AttrEmbedder[idx](edge_attr_s[:,idx+1]) for idx in range(self.num_attr)]
            attr_embs = torch.cat(embs, dim=1)  # Shape (num_edges, hidden_dim*num_attr)

            edge_emb = attr_embs + pos_vector  # Shape (num_edges, hidden_dim*num_attr)
            h_e = self.EdgeTransform(edge_emb)  # Shape (num_deges, hidden_dim)

        else:

            pos_vector = self.PosEnc(edge_attr_s[:,0])  # Shape (num_edges, hidden_dim)
            h_e = self.EdgeTransform(pos_vector)

        return h_x, h_e     # Shape (num_nodes, hidden_dim), (num_edges, hidden_dim)


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
        self.AttrEmbedder = nn.ModuleList([nn.Embedding(10000, hidden_dim) for _ in range(self.num_attr)])

        self.NodeTransform = nn.Sequential(
                             nn.Linear(int(hidden_dim*self.num_attr) if self.num_attr > 0 else hidden_dim, hidden_dim),
                             nn.LayerNorm(hidden_dim))          # Shape (num_nodes, hidden_dim)
        
        # For edges (activities)
        self.EdgeTransform = nn.Sequential(
                             nn.Linear(edge_dim, hidden_dim),
                             nn.LayerNorm(hidden_dim))          # Shape (num_edges, hidden_dim)

    def forward(self, x_t, edge_attr_t):
        # edges
        h_e = self.EdgeTransform(edge_attr_t) # Shape (num_edges, hidden_dim)

        if self.num_attr>0:

            pos_vector = self.PosEnc(x_t[:,0]).repeat(1, self.num_attr)

            embs = [self.AttrEmbedder[idx](x_t[:, idx+1]) for idx in range(self.num_attr)]
            attr_embs = torch.cat(embs, dim=1)  # Shape (num_edges, hidden_dim*num_attr)

            node_emb = attr_embs + pos_vector  # Shape (num_edges, hidden_dim*num_attr)

            h_x = self.NodeTransform(node_emb)  # Shape (num_nodes, hidden_dim)
        
        else:

            pos_vector = self.PosEnc(x_t[:,0]) # Shape (num_edges, hidden_dim)
            h_x = self.NodeTransform(pos_vector)

        return h_x, h_e     # Shape (num_nodes, hidden_dim), (num_edges, hidden_dim)


# For outcome prediction and anomaly detection
class DownstreamLayer(nn.Module):
    def __init__(self, latent_dim:int = 64, dropout:float = 0.3):
        super(DownstreamLayer, self).__init__()
        '''
        *Input: Shape(num_graphs, hidden_dim*2)
        '''
        self.linear1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.output = nn.Linear(latent_dim, 1)
    
    def forward(self, x):
        '''
        :param x: Shape(num_graphs, hidden_dim*2)
        '''
        x = self.linear1(x)  # Shape(num_graphs, hidden_dim)
        x = self.linear2(x)  # Shape(num_graphs, hidden_dim)
        x = self.output(x)   # Shape(num_graphs, 1)

        return x
