import os
from pathlib import Path
import pandas as pd
from utils.fs import EVENTLOG_DIR, ATTR_KEYS

from typing import Optional




class EventLog(object):

    def __init__(self, LogName:Optional[str]=None):

        if LogName is not None:
            LogName = os.path.splitext(LogName)[0]
            self.log = self.load(LogName)
            self.logname = LogName
        else: raise ValueError("LogName must be provided")
    
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