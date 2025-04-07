'''
@author: Y.J.Lee
'''


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

        if you want to train a new model,
        >>> from model.GCEL import GCEL
        >>> gcel = GCEL(*parameters)
        >>> gcel.fit(dataset, save=True, visualize=True)

        if you want to load a pre-trained model,
        >>> from model.GCEL import GCEL
        >>> gcel = GCEL()
        >>> gcel.load_model(logname)
        >>> gcel.get_embeddings(visualize=True)
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
        
        encoder.eval()
        embeddings = []
        for batch in loader:
            batch = batch.to(self.device)
            embs = encoder(batch, train=False)
            embeddings.append(embs.cpu().detach().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        return encoder, all_loss, embeddings

    def fit(self, dataset, save:bool = False, visualize:bool = False):
        '''
        Fit the model to the dataset.

        param dataset: The dataset object to be used for training.
        param save: If True, save the model parameters and loss plot.
        param visualize: If True, save the loss curve and TSNE visualization.
        '''

        self.logname = dataset.LogName
        self.graphs = dataset.graphs

        # Based on the first view
        self.ndim = self.graphs[0].x_s.shape[1]
        self.edim = self.graphs[0].edge_attr_s.shape[1]

        encoder, losses, embeddings = self.train()

        # Save the trained model and loss plot
        if save:
            torch.save({
                'params': encoder.state_dict(),
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'node_dim': self.ndim,
                'edge_dim': self.edim}, SAVE_DIR + f'/params/params_{self.logname}.pt')
            
            print(f"Model saved at {SAVE_DIR}/params/params_{self.logname}.pt")
        
        if visualize:
            plot_loss(losses, self.logname)
            plot_tsne(embeddings, self.logname)
            print(f"Loss plot saved at {SAVE_DIR}/loss/loss ({self.logname}).png")
            print(f"TSNE plot saved at {SAVE_DIR}/tsne/tsne ({self.logname}).png")

    def load_model(self, logname:str):
        '''
        Load the model from the saved path.
        >> logname: The name of the log.
        >> GCEL = GCEL()
        >> model = GCEL.load_model(logname)
        '''

        save_path = SAVE_DIR + f'/params/params_{logname}.pt'

        if not os.path.exists(save_path):
            raise ValueError(f"Model not found at {save_path}. Please train the model first.")

        checkpoint = torch.load(save_path, map_location=self.device)
        model = GraphEncoder(node_dim=checkpoint['node_dim'],
                             edge_dim=checkpoint['edge_dim'],
                             hidden_dim=checkpoint['hidden_dim'], 
                             num_layers=checkpoint['num_layers'], 
                             dropout=checkpoint['dropout']).to(self.device)
        
        model.load_state_dict(checkpoint['params'])
        
        for param in model.parameters():
            param.requires_grad = False

        return model

    def eval_clustering(self):
        pass

    def eval_outcome_pred(self):
        pass