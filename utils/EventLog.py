import pandas as pd
from utils.fs import EVENTLOG_DIR
import os
from pathlib import Path


class EventLog(object):

    def __init__(self, LogName):        
        self.log = self.load(LogName)
    
    def load(self, LogName):

        LogName = Path(LogName)
        if '.csv' not in LogName.suffixes:
            FileName = Path(str(LogName) + '.csv')
        FilePath = os.path.join(EVENTLOG_DIR, FileName)
        return pd.read_csv(FilePath)

    @property
    def cases(self):
        return [x for x in self.log['case_id'].unique()]
    
    @property
    def case_len(self):
        pass

    @property
    def max_case_len(self):
        pass

    @property
    def feature(self):
        pass
