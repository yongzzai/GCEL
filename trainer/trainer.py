'''
@author: Y.J.Lee
'''

from .variantSampler import NegativeSampler
from .loss import InfoNCE

def Trainer(epochs, encoder, loader, device, optimizer, scheduler) -> list:

    sampler = NegativeSampler(pad_mode='max')
    criterion = InfoNCE(temperature=0.1, negative_mode='paired')

    all_loss = []

    for param in encoder.parameters():
        param.requires_grad = True

    for epoch in range(epochs):
        encoder.train()
        epoch_loss = 0.0

        for batch in loader:
            batch = batch.to(device)

            _, _, dense_z1, dense_z2 = encoder(batch, train=True)
            negatives = sampler(dense_z1, batch.varlabel)       # Shape(batch_size, M, hidden)
            
            loss = criterion(dense_z1, dense_z2, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        all_loss.append(epoch_loss/len(loader)) # for visualization

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}')

    for param in encoder.parameters():
        param.requires_grad = False

    return all_loss