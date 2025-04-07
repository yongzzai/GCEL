'''
@author: Y.J.Lee
'''

import matplotlib.pyplot as plt
from utils.fs import SAVE_DIR
from sklearn.manifold import TSNE


import os
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
    
    plt.savefig(SAVE_DIR + f'/figs/loss ({logname}).png', dpi=600)


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
    plt.savefig(SAVE_DIR + f'/figs/TSNE ({logname}).png', dpi=600)
