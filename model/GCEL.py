import torch
import numpy as np

from .loss import InfoNCE
from .layers import GraphEncoder
from torch_geometric.loader import DataLoader
from .variantSampler import NegativeSampler
from utils.fs import SAVE_DIR
from logger.visualizer import *

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

        optimizer = torch.optim.AdamW(encoder.parameters(), lr=self.lr, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs, eta_min=0)
        
        sampler = NegativeSampler(pad_mode='max')
        criterion = InfoNCE(temperature=0.1, negative_mode='paired')

        for param in encoder.parameters():
            param.requires_grad = True
        
        all_loss = []
    
        for epoch in range(self.epochs):

            encoder.train()
            epoch_loss = 0.0

            for batch in loader:
                batch = batch.to(self.device)

                _, _, dense_z1, dense_z2 = encoder(batch, train=True)
                negatives = sampler(dense_z1, batch.varlabel)       # Shape(batch_size, M, hidden)
                
                loss = criterion(dense_z1, dense_z2, negatives)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            all_loss.append(epoch_loss/len(loader)) # for visualization

            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(loader):.4f}')
        
        for param in encoder.parameters():
            param.requires_grad = False
        
        return encoder, all_loss


    def fit(self, dataset):
        
        self.graphs = dataset.graphs

        # Based on the first view
        self.ndim = self.graphs[0].x_s.shape[1]
        self.edim = self.graphs[0].edge_attr_s.shape[1]

        encoder, losses = self.train()

        self.logname = dataset.LogName

        self.save_path = SAVE_DIR + f'/params/Enc ({self.logname}).pt'
        torch.save({
            'params': encoder.state_dict(),
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'node_dim': self.ndim,
            'edge_dim': self.edim}, self.save_path)
                
        plot_loss(losses, self.logname)

    def load(self):

        checkpoint = torch.load(self.save_path, map_location=self.device)
        model = GraphEncoder(node_dim=checkpoint['node_dim'],
                             edge_dim=checkpoint['edge_dim'],
                             hidden_dim=checkpoint['hidden_dim'], 
                             num_layers=checkpoint['num_layers'], 
                             dropout=checkpoint['dropout']).to(self.device)
        
        model.load_state_dict(checkpoint['params'])
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model


    def visualize(self):

        model = self.load()
        
        loader = DataLoader(self.graphs, batch_size=self.batch_size, shuffle=False, follow_batch=['x_s','x_t'])
        
        embeddings = []
        for batch in loader:
            batch = batch.to(self.device)

            emb = model(batch, train=False)
            embeddings.append(emb.cpu().detach().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
    
        TSNEembs(embeddings, self.logname)