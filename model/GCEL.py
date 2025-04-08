'''
@author: Y.J.Lee
'''


import torch
import numpy as np
from torch_geometric.loader import DataLoader


from .layers import GraphEncoder
from utils.fs import SAVE_DIR
from logger.visualizer import *
from downstream.Clustering import DS_Clustering
from trainer.trainer import Trainer

class GCEL:

    def __init__(self,
                 hidden_dim:int = 64, num_layers:int = 1, dropout:float = 0.3,
                 epochs:int = 20, lr:float = 0.001, batch_size:int = 128):
        
        '''
        Graph Contrastive Event log Learning (GCEL)
        
        :param hidden_dim: The dimension of the hidden.
        :param num_layers: The number of layers.
        :param dropout: The dropout rate.
        :param epochs: The number of epochs.
        :param lr: The learning rate.
        :param batch_size: The batch size.

        >>> from model.GCEL import GCEL
        >>> gcel = GCEL(*parameters)
        >>> gcel.fit(dataset, save=True, visualize=True)
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
        
        all_loss = Trainer(self.epochs, encoder, loader, self.device, optimizer, scheduler)
        
        embeddings = self.get_embs(loader, encoder)
        
        return encoder, all_loss, embeddings


    def get_embs(self, loader, encoder):
        encoder.eval()
        embeddings = []
        for batch in loader:
            batch = batch.to(self.device)
            embs = encoder(batch, train=False)
            embeddings.append(embs.cpu().detach().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

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


    def clustering(self):
        model = self.load_model(self.logname)
        model.eval()
        
        nmi, ari, _, _ = DS_Clustering(model, self.graphs, self.device)
        print(f"Normalized Mutual Information: {nmi:.4f},\n Adjusted_Rand_Score: {ari:.4f}")

        #TODO: 클러스터링 결과 시각화 추가

    def eval_outcome_pred(self):
        pass