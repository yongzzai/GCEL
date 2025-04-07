'''
@author: Y.J.Lee
'''

from .PairData import PairData
from .augmGraphs import genEventGraph, genActivityGraph
import ray

@ray.remote
def GenerateGraphs(splitlog, onehot_dict, event_attrs) -> list:
    # /*create a graph for each case
    # 1st graph = activity as node, positional encoding + attr as edge
    # 2nd graph = events as node, activity as edge
    # >> transpose (node -> edge, edge -> node) 1st graph == 2nd graph
    
    PairGraphs = []

    for _, caseid in enumerate(splitlog['case_id'].unique()):

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

        # Get Cluster Ground Truth
        if 'clabel' in sublog.columns:
            cluster_label = sublog['clabel'].values[0]

            GraphPair = PairData(x_s=g1.x, edge_index_s=g1.edge_index, edge_attr_s=g1.edge_attr,
                                  x_t=g2.x, edge_index_t=g2.edge_index, edge_attr_t=g2.edge_attr,
                                  varlabel = int(variant), caseid=caseid, clabel=int(cluster_label))
        else:
            GraphPair = PairData(x_s=g1.x, edge_index_s=g1.edge_index, edge_attr_s=g1.edge_attr,
                                  x_t=g2.x, edge_index_t=g2.edge_index, edge_attr_t=g2.edge_attr,
                                  varlabel = int(variant), caseid=caseid)
        
        
        PairGraphs.append(GraphPair)
    
    return PairGraphs
