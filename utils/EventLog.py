import pandas as pd
from utils.directory import EVENTLOG_DIR
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