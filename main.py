'''
@author: Y.J.Lee
'''


import argparse

#parser = argparse.ArgumentParser(description='run')
#parser.add_argument('-r','--runtype', type=str, required=True, help="one of -> 'create_graphs', 'train', 'test', 'predict'")
#args = parser.parse_args()

from dsprovider.dataset import Dataset
dd = Dataset('BPIC12')



#* For reviewing the graph augmentation modules
cid = dd.graphs[1].caseid
anchorlog = dd.EventLog.log[dd.EventLog.log['case_id'] == cid].copy()
print(anchorlog)
print(dd.graphs[1])
print(dd.graphs[1].x_s)
print(dd.graphs[1].edge_index_s)
print(dd.graphs[1].edge_attr_s)

print("====================================")
print(dd.graphs[1])
print(dd.graphs[1].x_t)
print(dd.graphs[1].edge_index_t)
print(dd.graphs[1].edge_attr_t)

