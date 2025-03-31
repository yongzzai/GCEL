'''
@author: Y.J.Lee
'''

from .augmGraphs import genEventGraph, genActivityGraph
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

        # /*1st graph (Activity as node)
        g1 = genActivityGraph(sublog, onehot_dict, event_attrs)      # View 1

        # /*2nd graph (Event as node)
        g2 = genEventGraph(g1)                                       # View 2

        # Get Variant Label
        variant = sublog['@variant'].values[0]

        GraphMatch = PairData(x_s=g1.x, edge_index_s=g1.edge_index, edge_attr_s=g1.edge_attr,
                x_t=g2.edge_attr, edge_index_t=g2.edge_attr, edge_attr_t=g2.edge_attr,
                varlabel = int(variant), caseid=caseid)

        PairGraphs.append(GraphMatch)
    
    return PairGraphs
