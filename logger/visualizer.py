'''
@author: Y.J.Lee
'''


import os
import numpy as np
import matplotlib.pyplot as plt
from utils.fs import SAVE_DIR
from sklearn.manifold import TSNE


os.environ['KMP_DUPLICATE_LIB_OK']='True'


def plot_loss(losses, logname):
    '''
    Plot the loss values.
    '''
    plt.figure(figsize=(10, 5))
    epochs = list(range(1, len(losses) + 1))
    plt.plot(epochs, losses, label='Loss', color='blue')
    plt.xticks(epochs)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid()
    
    plt.savefig(SAVE_DIR + f'/loss/loss ({logname}).png', dpi=600)


def plot_tsne(embs, logname):
    '''
    Plot the t-SNE embeddings.
    '''

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embs)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=2, alpha=0.5)
    plt.title('t-SNE Visualization of Embeddings')
    plt.grid()
    plt.savefig(SAVE_DIR + f'/tsne/TSNE ({logname}).png', dpi=600)


def plot_clusters(preds, labels, embeddings, logname):
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.join(SAVE_DIR, 'ds_cluster'), exist_ok=True)
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    # Get unique values
    unique_preds = np.unique(preds)
    unique_labels = np.unique(labels)
    
    # Warm colors for predictions (reds, oranges, yellows)
    warm_cmap = plt.cm.Reds_r
    # Cool colors for labels (blues, greens)
    cool_cmap = plt.cm.Blues_r
    
    # Plot and save predicted clusters with warm colors
    plt.figure(figsize=(10, 8))
    for i, cluster in enumerate(unique_preds):
        mask = preds == cluster
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                   color=warm_cmap(i/len(unique_preds)), 
                   label=f'Cluster {cluster}',
                   alpha=0.7, s=3, edgecolors='w', linewidth=0.5)
    
    plt.title(f'Predicted Clusters', fontsize=16)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save predictions figure
    pred_save_path = os.path.join(SAVE_DIR, f'ds_cluster/pred_clustering_{logname}.png')
    plt.savefig(pred_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot and save ground truth labels with cool colors
    plt.figure(figsize=(10, 8))
    for i, cluster in enumerate(unique_labels):
        mask = labels == cluster
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                   color=cool_cmap(i/len(unique_labels)), 
                   label=f'Label {cluster}',
                   alpha=0.7, s=3, edgecolors='w', linewidth=0.5)
    
    plt.title(f'Ground Truth Clusters', fontsize=16)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save ground truth figure
    label_save_path = os.path.join(SAVE_DIR, f'ds_cluster/true_clustering_{logname}.png')
    plt.savefig(label_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Clustering visualizations saved at {os.path.dirname(pred_save_path)}")