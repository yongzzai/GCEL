'''
@author: Y.J.Lee
'''


from pm4py import split_by_process_variant

def GetVariantLabels(log):
    '''
    Get variant labels from event log
    '''
    vardict = split_by_process_variant(log, activity_key='name', case_id_key='case_id', timestamp_key='timestamp')
    
    for idx, (_, sublog) in enumerate(vardict):
        log.loc[log['case_id'].isin(sublog['case_id'].unique()), '@variant'] = int(idx)

    return log
