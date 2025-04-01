'''
@Author: Y.J.Lee
'''

from torch_geometric.loader import DataLoader

def get_loader(dataset, batch_size, shuffle, pin_memory):
    '''
    This function is used to create a dataloader for the dataset.
    
    :param dataset: The dataset to be loaded.
    :param batch_size: The batch size of the dataloader.
    :param shuffle: Whether to shuffle the dataset or not.
    :param pin_memory: Whether to pin memory or not.
    :return: The dataloader for the dataset.
    '''
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

