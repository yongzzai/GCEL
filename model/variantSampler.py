import torch

class NegativeSampler(torch.nn.Module):

    '''
    Sampling negatives from the embeddings of the graph in the same batch.
    The negatives are sampled based on the labels of the variants.

    *Args
        :param pad_mode: 'min' or 'max'
        - 'min': undersample the negative samples to make them equal to the min count.
        - 'max': oversample the negative samples to make them equal to the max count.
    
    *Input
        :embs: graph embeddings with Shape(batch_size, dim)
        :label: variant labels with Shape(batch_size)
    
    *Output
        :negative embs with Shape(batch_size, num_negatives, hidden)
    '''

    def __init__(self, pad_mode: str = 'max'):
        super(NegativeSampler, self).__init__()

        if pad_mode not in ['min', 'max']:
            raise ValueError("pad_mode must be either 'min' or 'max'")

        self.pad_mode = pad_mode
    
    def forward(self, embs, label):

        label = label.reshape(-1, 1)    # Shape([batch_size, 1])

        if self.pad_mode == 'max':

            # Get the unique labels and their counts
            max_count = 0
            for lab in torch.unique(label):
                neg_count = (label != lab).sum().item()
                max_count = max(max_count, neg_count)

            # print(max_count)

            NegChunks = []

            for idx in range(embs.shape[0]):

                anc_label = label[idx].item()

                # embeddings corresponding to idx with labels different from anc_label
                neg_indices = torch.where(label != anc_label)[0]
                neg_embs = embs[neg_indices]

                if neg_embs.shape[0] < max_count:
                    # padding by oversampling randomly.
                    neg_embs = torch.cat([neg_embs, neg_embs[torch.randint(0, neg_embs.shape[0], (max_count - neg_embs.shape[0],))]])

                    NegChunks.append(neg_embs)
    
                else:   # if neg_embs.shape[0] == max_count:
                    NegChunks.append(neg_embs)
                                    
            negative_samples = torch.stack(NegChunks)   # Shape([batch_size, max_count, dim])

            # print(negative_samples.shape)

            return negative_samples
        
        elif self.pad_mode == 'min':

            min_count = 1e+10
            for lab in torch.unique(label):
                neg_count = (label != lab).sum().item()
                min_count = min(min_count, neg_count)
            
            # print(min_count)

            NegChunks = []

            for idx in range(embs.shape[0]):

                anc_label = label[idx].item()

                # embeddings corresponding to idx with labels different from anc_label
                neg_indices = torch.where(label != anc_label)[0]
                neg_embs = embs[neg_indices]

                if neg_embs.shape[0] > min_count:
                    # padding by undersampling randomly.
                    neg_embs = neg_embs[torch.randperm(neg_embs.shape[0])[:min_count]]

                    NegChunks.append(neg_embs)
    
                else:
                    NegChunks.append(neg_embs)
            
            negative_samples = torch.stack(NegChunks)   # Shape([batch_size, min_count, dim])

            # print(negative_samples.shape)

            return negative_samples
