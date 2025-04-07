'''
@author: Y.J.Lee
'''

import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
EVENTLOG_DIR = os.path.join(ROOT_DIR,'eventlogs')
SAVE_DIR = os.path.join(ROOT_DIR, 'saved')

ATTR_KEYS = {
    'BPIC12':{'AttributeKeys':['name']},
    
    'BPIC13_closed_problems':{'AttributeKeys':['name', 'org_group', 'org_resource', 'organization_country', 'resource_country']},
    'BPIC13_incidents':{'AttributeKeys':['name', 'org_group', 'org_resource', 'organization_country', 'resource_country']},
    'BPIC13_open_problems':{'AttributeKeys':['name', 'org_group', 'org_resource']},
    'BPIC13c':{'AttributeKeys':['name', 'org_group', 'org_resource']},
    
    'BPIC15_1':{'AttributeKeys':['name', 'monitoringResource', 'org_resource']},
    'BPIC15_2':{'AttributeKeys':['name', 'monitoringResource', 'org_resource']},
    'BPIC15_3':{'AttributeKeys':['name', 'action_code', 'monitoringResource', 'org_resource']},
    'BPIC15_4':{'AttributeKeys':['name', 'monitoringResource', 'org_resource']},
    'BPIC15_5':{'AttributeKeys':['name', 'monitoringResource', 'org_resource']},
    'BPIC15c':{'AttributeKeys':['name', 'monitoringResource', 'org_resource']},

    'BPIC17':{'AttributeKeys':['name', 'org_resource']},
    'BPIC17_offer':{'AttributeKeys':['name', 'org_resource']},
    'BPIC17c':{'AttributeKeys':['name', 'org_resource']},

    'BPIC20_DomesticDeclarations':{'AttributeKeys':['name', 'org_resource', 'org_role']},
    'BPIC20_InternationalDeclarations':{'AttributeKeys':['name', 'org_resource', 'org_role']},
    'BPIC20_PermitLog':{'AttributeKeys':['name', 'org_resource', 'org_role']},
    'BPIC20_PrepaidTravelCost':{'AttributeKeys':['name', 'org_resource', 'org_role']},
    'BPIC20_RequestForPayment':{'AttributeKeys':['name', 'org_resource', 'org_role']},
    'BPIC20c':{'AttributeKeys':['name', 'org_resource', 'org_role']},

    'Receipt':{'AttributeKeys':['name', 'org_group', 'org_resource']},
    'Sepsis':{'AttributeKeys':['name','org_group']},

    'Road_Traffic_Fine_Management_Process':{'AttributeKeys':['name']},
    'Billing':{'AttributeKeys':['name']}
}