'''
@author: Y.J.Lee
'''


import torch
import numpy as np
from torch_geometric.data import Data


def genActivityGraph(sublog, onehot_dict, event_attrs) -> Data:

    num_nodes = sublog['activity_order'].nunique() + 1 # synthetic start node
    x = np.zeros((num_nodes, len(onehot_dict)))
    
    for _, act_order in enumerate(sublog['activity_order'].unique()):
        x[act_order] = list(sublog[sublog['activity_order']==act_order]['onehot'].values[0])

    x = torch.tensor(x, dtype=torch.float)

    ### Edges
    edges = [(0, sublog.loc[0,'activity_order'])] + [
        [sublog['activity_order'][idx0], sublog['activity_order'][idx0+1]]
                        for idx0 in range(len(sublog)-1)]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    ### Edge Attributes
    if len(event_attrs)>0:
        selected_cols = ['event_position'] + event_attrs
        edge_attr = np.array([sublog.loc[idx, selected_cols] for idx in range(len(sublog))])
        edge_attr = edge_attr.astype(np.int16)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    else:
        edge_attr = np.array([sublog.loc[idx, 'event_position'] for idx in range(len(sublog))])
        edge_attr = edge_attr.astype(np.int16)
        edge_attr = edge_attr.reshape(-1, 1)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    view1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return view1




# /* view1 --> view2 
def genEventGraph(view1):

    orig_edge_index = view1.edge_index
    num_edges = orig_edge_index.size(1)
    new_edges = []         
    new_edge_attrs = []
    
    # Create a map to store event positions
    event_positions = {}
    for i in range(num_edges):
        # Store the position for each edge in the original graph
        # Event position is stored in the first dimension of edge_attr
        event_positions[i] = view1.edge_attr[i, 0].item()
    
    num_nodes = view1.num_nodes if hasattr(view1, 'num_nodes') else view1.x.size(0)
    
    # For every original nodes, considering the combination of incoming, outgoing edges
    for v in range(num_nodes):
        # edge index (u,v) ends at node v
        incoming_mask = (orig_edge_index[1] == v)
        incoming_indices = torch.nonzero(incoming_mask, as_tuple=False).view(-1)
        
        # edge index (v,w) start from node v
        outgoing_mask = (orig_edge_index[0] == v)
        outgoing_indices = torch.nonzero(outgoing_mask, as_tuple=False).view(-1)
        
        for i in incoming_indices:
            i_pos = event_positions[i.item()]
            for j in outgoing_indices:
                j_pos = event_positions[j.item()]
                # Only connect if j event occurs after i event (forward direction)
                if j_pos > i_pos:
                    new_edges.append([i.item(), j.item()])
                    new_edge_attrs.append(view1.x[v])
    
    if new_edges:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        new_edge_attr = torch.stack(new_edge_attrs, dim=0)
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_edge_attr = torch.empty((0, view1.x.size(1)))
    
    # original edge feature == new node feature
    if hasattr(view1, 'edge_attr') and view1.edge_attr is not None:
        new_x = view1.edge_attr
    else:
        new_x = torch.eye(num_edges)
    
    view2 = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)

    return view2