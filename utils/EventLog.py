'''
@author: Y.J.Lee
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
from utils.fs import EVENTLOG_DIR, ATTR_KEYS

from typing import Optional


class EventLog(object):

    def __init__(self, LogName:Optional[str]=None):

        if LogName is not None:
            LogName = os.path.splitext(LogName)[0]
            self.log = self.load(LogName)
            self.logname = LogName
        else: raise ValueError("LogName must be provided")

        #/* remove padding events
        self.log = self.log[~self.log['name'].isin(['▶', '■'])]

        # /*event position start from 0
        self.log['event_position'] = self.log['event_position'] - 1

        # /*set datatype
        self.log['case_id'] = self.log['case_id'].astype(str)
        self.log['name'] = self.log['name'].astype(str)
        self.log['timestamp'] = self.log['timestamp'].apply(lambda x: str(x).split('+')[0].replace('T', ' '))
        self.log['timestamp'] = pd.to_datetime(self.log['timestamp'])
        
    def load(self, LogName):
        FileName = Path(str(LogName) + '.csv')
        FilePath = os.path.join(EVENTLOG_DIR, FileName)
        return pd.read_csv(FilePath)
    

    ### Case Properties
    @property
    def caseids(self) -> list:
        return self.log['case_id'].unique().tolist()
    
    @property
    def max_trace_len(self) -> int:
        return max(self.log['event_position'])-1

    ### Event Attribute Properties
    @property
    def attr_keys(self) -> list:
        return ATTR_KEYS[self.logname]['AttributeKeys']
    
    @property
    def num_attr(self) -> int:
        return len(self.attr_keys)
    
    @property
    def num_uniq_activity(self) -> int:
        return self.log['name'].nunique()
    
    @property
    def onehot_dictionary(self) -> dict:
        return {act: vector for act, vector in zip(self.log['name'].unique(), np.eye(self.log['name'].nunique()))}

    @property
    def event_attrs(self) -> list:
        attrs = ATTR_KEYS[self.logname]['AttributeKeys'].copy()
        if 'name' in attrs:
            attrs.remove('name')
        return attrs