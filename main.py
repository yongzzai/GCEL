'''
@author: Y.J.Lee
'''


import argparse

#parser = argparse.ArgumentParser(description='run')
#parser.add_argument('-r','--runtype', type=str, required=True, help="one of -> 'create_graphs', 'train', 'test', 'predict'")
#args = parser.parse_args()


from modules.dataset import Dataset
dd = Dataset('BPIC12')
# print(dd.graphs[0])
# print(dd.graphs[0].x_s)
# print(dd.graphs[0].edge_index_s)
# print(dd.graphs[0].edge_attr_s)

# print(dd.graphs[1])
# print(dd.graphs[1].x_s)
# print(dd.graphs[1].edge_index_s)
# print(dd.graphs[1].edge_attr_s)
# # from utils.EventLog import EventLog
# # el = EventLog('BPIC12')
# # print(el.log.columns)
# # print(el.onehot_dictionary)

