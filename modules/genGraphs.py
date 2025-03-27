'''
@author: Y.J.Lee
'''

from .augm import AugmentGraph
import torch
import numpy as np
from torch_geometric.data import Data
import ray



# /*PairData can contain two pyg Data objects concurrently
class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


@ray.remote
def GenerateGraphs(splitlog, onehot_dict, event_attrs) -> list:
    # /*create a graph for each case
    # 1st graph = activity as node, positional encoding + attr as edge
    # 2nd graph = events as node, activity as edge
    # >> transpose (node -> edge, edge -> node) 1st graph == 2nd graph
    
    PairGraphs = []

    for cidx, caseid in enumerate(splitlog['case_id'].unique()):

        sublog = splitlog[splitlog['case_id']==caseid]
        sublog = sublog.reset_index(drop=True)
        subUniqActivity = sublog['name'].unique()
        actOrderMap = {act: idx+1 for idx, act in enumerate(subUniqActivity)}
        
        ### activity order (for connecting nodes)
        sublog['activity_order'] = sublog['name'].map(actOrderMap)
        sublog['onehot'] = sublog['name'].map(onehot_dict)

        # /*1st graph (Activity as node) -> directly follows graph style

        ### Node Features (onehot acitivities)
        num_nodes = sublog['activity_order'].nunique() + 1 # synthetic start node
        x = np.zeros((num_nodes, len(onehot_dict)))        # syn-node remain zero

        for idx, act_order in enumerate(sublog['activity_order'].unique()):
            x[act_order] = list(sublog[sublog['activity_order']==act_order]['onehot'].values[0])

        # x[sublog['activity_order'].values] = sublog['onehot'].values
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
        else:
            edge_attr = np.array([sublog.loc[idx, 'event_position'] for idx in range(len(sublog))])
        edge_attr = edge_attr.astype(np.int16)

        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        # Get Variant Label
        variant = sublog['@variant'].values[0]

        g1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # /*2nd graph (Event as node)
        g2 = AugmentGraph(g1)

        GraphMatch = PairData(x_s=g1.x, edge_index_s=g1.edge_index, edge_attr_s=g1.edge_attr,
                x_t=g2.edge_attr, edge_index_t=g2.edge_attr, edge_attr_t=g2.edge_attr,
                varlabel = int(variant), caseid=caseid)

        PairGraphs.append(GraphMatch)
    
    return PairGraphs
