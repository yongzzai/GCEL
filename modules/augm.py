'''
@author: Y.J.Lee
'''


import torch
from torch_geometric.data import Data

def AugmentGraph(data: Data) -> Data:
    """
    original: u --[Edge(u, v)]-- v --[Edge(v, w)]-- w
    new: Node(u,v) -> Edge(v) -> Node(v,w)
    """

    orig_edge_index = data.edge_index
    num_edges = orig_edge_index.size(1)
    new_edges = []         
    new_edge_attrs = []
    
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
    
    # For every original nodes, considering the combination of incoming, outgoing edges
    for v in range(num_nodes):
        # edge index (u,v) ends at node v
        incoming_mask = (orig_edge_index[1] == v)
        incoming_indices = torch.nonzero(incoming_mask, as_tuple=False).view(-1)
        
        # edge index (v,w) start from node v
        outgoing_mask = (orig_edge_index[0] == v)
        outgoing_indices = torch.nonzero(outgoing_mask, as_tuple=False).view(-1)
        
        for i in incoming_indices:
            for j in outgoing_indices:
                new_edges.append([i.item(), j.item()])
                new_edge_attrs.append(data.x[v])
    
    if new_edges:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        new_edge_attr = torch.stack(new_edge_attrs, dim=0)
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_edge_attr = torch.empty((0, data.x.size(1)))
    
    # original edge feature == new node feature
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        new_x = data.edge_attr
    else:
        new_x = torch.eye(num_edges)
    
    new_data = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
    return new_data
