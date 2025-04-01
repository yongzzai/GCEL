import torch
from loss import InfoNCE
from layers import GraphEncoder
from torch_geometric.loader import DataLoader

class GCEL:

    def __init__(self,
                 hidden_dim:int = 64, num_layers:int = 1, dropout:float = 0.3,
                 epochs:int = 20, lr:float = 0.001, batch_size:int = 128):
        
        '''
        Graph Contrastive Event log Learning (GCEL) Framework
        
        :param hidden_dim: The dimension of the hidden.
        :param num_layers: The number of layers.
        :param dropout: The dropout rate.
        :param epochs: The number of epochs.
        :param lr: The learning rate.
        :param batch_size: The batch size.
        '''

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train(self):
        
        encoder = GraphEncoder(node_dim=self.ndim, 
                               edge_dim=self.edim, 
                               hidden_dim=self.hidden_dim, 
                               num_layers=self.num_layers, 
                               dropout=self.dropout).to(self.device)
        
        loader = DataLoader(self.graphs, batch_size=self.batch_size, shuffle=True, follow_batch=['x_s','x_t'])

        optimizer = torch.optim.AdamW(encoder.parameters(), lr=self.lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs, eta_min=0)
        
        criterion = InfoNCE(temperature=0.1).to(self.device)

        for param in encoder.parameters():
            param.requires_grad = True
    
        for epoch in range(self.epochs):

            encoder.train()
            total_loss = 0.0

            for batch in loader:

                batch = batch.to(self.device)

                #TODO


    def fit(self, dataset):
        
        self.graphs = dataset.graphs

        # Based on the first view
        self.ndim = self.graphs[0].x_s.shape[1]
        self.edim = self.graphs[0].edge_attr_s.shape[1]

        self.encoder = self.train()
















