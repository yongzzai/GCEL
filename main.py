import argparse

#parser = argparse.ArgumentParser(description='run')
#parser.add_argument('-r','--runtype', type=str, required=True, help="one of -> 'create_graphs', 'train', 'test', 'predict'")
#args = parser.parse_args()


from utils.EventLog import EventLog
dd = EventLog('BPIC12')
print(dd.attr_keys)