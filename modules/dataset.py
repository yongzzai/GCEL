'''
@author: Y.J.Lee
'''

import os
import ray
import time
import pickle
import numpy as np
from itertools import chain

from utils.EventLog import EventLog
from .variants import GetVariantLabels
from .genGraphs import GenerateGraphs

class Dataset(object):

    def __init__(self, LogName):

        PrjPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        CachePath = os.path.join(PrjPath, 'eventlogs/cache')

        os.makedirs(CachePath, exist_ok=True)
        
        self.EventLog = EventLog(LogName)
        self.CacheFile = os.path.join(CachePath, f'Graph_{self.EventLog.logname}.pkl.gz')
        
        if not os.path.exists(self.CacheFile):
            print("Cached graphs are not detected, Creating graphs...")
            self.ProcessLog()   #/*preprocess event_log
            self.ConvertLog()   #/*convert log to graphs
            print(f"Graphs created and saved to {self.CacheFile}")
        
        else:
            print("Cached graphs are detected, Loading graphs...")
            with open(self.CacheFile, 'rb') as f:
                self.graphs = pickle.load(f)
            print("Graphs loaded")


    def ProcessLog(self):

        assert 'name' in self.EventLog.log.columns, "'name' column is missing in the dataset"
        assert 'event_position' in self.EventLog.log.columns, "'event_position' column is missing in the dataset"
        assert 'case_id' in self.EventLog.log.columns, "'case_id' column is missing in the dataset"

        # /*simplify column name
        self.EventLog.log.rename(columns=lambda x: x.replace(':','_').replace(' ','_'), inplace=True)

        # /*Set Variant Label
        self.EventLog.log = GetVariantLabels(self.EventLog.log)

        # /*label encoding attrs
        if len(self.EventLog.event_attrs) > 0:
            for key in self.EventLog.event_attrs:
                unique_values = set(self.EventLog.log[key])
                label_map = {label:idx+1 for idx, label in enumerate(unique_values)}
                self.EventLog.log[key] = self.EventLog.log[key].map(label_map)
        
        else: pass

    #/* for parallel processing
    def SplitLog(self) -> dict:

        NumCore = os.cpu_count() - 2    # 2 for web surfing 😀
        print(f"Number of CPU Cores: {NumCore}")

        CaseChunks = np.array_split(self.EventLog.caseids, NumCore)
        
        SeperateLogs = {}
        for idx, cases in enumerate(CaseChunks):
            SeperateLogs[idx] = self.EventLog.log[self.EventLog.log['case_id'].isin(cases)]
        
        return SeperateLogs

    def ConvertLog(self):

        SeparateLog = self.SplitLog()

        start_time = time.time()

        ProcessedLog = []
        onehot_dict = self.EventLog.onehot_dictionary
        event_attrs = self.EventLog.event_attrs

        ray.init()
        [ProcessedLog.append(GenerateGraphs.remote(splitlog, onehot_dict, event_attrs)) for splitlog in SeparateLog.values()]
        graphs = ray.get(ProcessedLog)
        ray.shutdown()

        self.graphs = list(chain.from_iterable(graphs))

        with open(self.CacheFile, 'wb') as f:
            pickle.dump(self.graphs, f)

        end_time = time.time()
        print("Time elapsed: ", end_time - start_time)
        
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

