from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import rand_score


def evaluate_clustering(label, preds):
    '''
    Evaluate clustering performance using NMI and ARI.
    >> true_labels: The ground truth labels.
    >> predicted_labels: The predicted labels.
    '''
    nmi = normalized_mutual_info_score(label, preds)
    ari = rand_score(label, preds)
    return nmi, ari


