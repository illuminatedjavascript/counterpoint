import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from progress.bar import Bar


class Trainer():
    """Trainer class for ChoraleBertModel"""
    def __init__(self, model, dataset, lr):
        self.model = model
        self.dataset = dataset
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self, num_epochs, batch_size):
        loader = DataLoader(self.dataset, batch_size, shuffle=True)
        for epoch in range(num_epochs):
            loss = self._train_epoch(loader)
            print(f'Epoch {epoch+1} / {num_epochs} complete:')
            print(f'Train loss = {loss}.')
            print(f'Test loss = {self.test()}.')
            
    def _train_epoch(self, loader):
        """Trains epoch & returns loss for last batch."""
        self.model.train()

        with Bar('Training epoch...', max=len(loader)) as bar:
            for src, tgt in loader:
                pred = self.model.forward(src)
                # nn.CrossEntropyLoss requires pred shape (#batches, #classes, seq_len)
                loss = self.loss_fn(pred.transpose(1, 2), tgt) 
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                bar.next()
        
        return loss.item()
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        
        src, tgt = self.dataset.get_test(100)
        pred = self.model.forward(src)
        loss = self.loss_fn(pred.transpose(1, 2), tgt) 
        
        return loss.item()