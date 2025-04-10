'''
@author: Y.J.Lee
'''

from torch_geometric.loader import DataLoader
from sklearn.cluster import KMeans
from utils.eval import evaluate_clustering
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# to cluster the embeddings using KMeans
def DS_Clustering(model, graphs, device):
    '''
    Clustering the embeddings using KMeans.
    '''
    clabels = []
    embeddings = []
    loader = DataLoader(graphs, batch_size=128, shuffle=False, follow_batch=['x_s','x_t'])
    for graph in loader:
        graph = graph.to(device)
        embs = model(graph, train=False)
        embeddings.append(embs.cpu().detach().numpy())
        clabels.append(graph.clabel.cpu().detach().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    clabels = np.concatenate(clabels, axis=0)

    kmeans = KMeans(n_clusters=len(np.unique(clabels)), random_state=42).fit(embeddings)
    preds = kmeans.labels_
    clabels = clabels.flatten()
    nmi, ari = evaluate_clustering(clabels, preds)

    return nmi, ari, preds, clabels, embeddings