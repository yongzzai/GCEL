'''
@author: Y.J.Lee
'''

import torch

def maskNodes(x, batch_idx, p:float = 0.2):
    """
    Randomly dropping node feature in each graph of a batch.
    
    param x: Node features tensor of shape [num_nodes, feature_dim]
    param batch_idx: Batch assignment tensor of shape [num_nodes]
    param p: Probability of masking a node (fraction of nodes to mask)
    """
    mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    
    # Process each graph in the batch separately
    for i in range(batch_idx.max().item() + 1):
        # Find nodes belonging to this graph
        graph_mask = batch_idx == i
        graph_nodes = graph_mask.sum().item()
        
        num_to_mask = round(graph_nodes * p)
        num_to_mask = num_to_mask if num_to_mask >= 1 else 1
        
        if num_to_mask > 0:
            # Get indices of nodes in this graph
            node_indices = torch.nonzero(graph_mask, as_tuple=True)[0]
            
            # Randomly select indices to mask
            perm = torch.randperm(graph_nodes)
            mask_indices = node_indices[perm[:num_to_mask]]
            
            # Set the mask for these indices
            mask[mask_indices] = True
     
    # matching shape of x
    mask = mask.unsqueeze(1).expand_as(x)    

    x[mask] = 0

    return x




