import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from progress.bar import Bar
from .model import ChoraleBertModel
from .dataset import ChoraleDataset


class Trainer():
    """Trainer class for ChoraleBertModel.
    Args:
        model: ChoraleBertModel instance to be trained.
        dataset: ChoreDataset instance to train on.
        lr: learning rate for training.
    """
    def __init__(self, model: ChoraleBertModel, dataset: ChoraleDataset, lr: float):
        self.model = model
        self.dataset = dataset
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self, num_epochs: int, batch_size: int):
        """Train loop for BertChoraleModel.
        Args:
            num_epochs: number of epochs to train for.
            batch_size: size of each batch.
        """
        loader = DataLoader(self.dataset, batch_size, shuffle=True)
        for epoch in range(num_epochs):
            loss = self._train_epoch(loader)
            print(f'Epoch {epoch+1} / {num_epochs} complete:')
            print(f'Last batch train loss = {loss}.')
            print(f'Test loss = {self.test_loss()}.')
            
    def _train_epoch(self, loader):
        """Trains epoch & returns loss for last batch.
        Args:
            loader: torch DataLoader instance for batching.
        Returns:
            loss: loss model calculated on the last batch of the epoch.
        """
        self.model.train()

        with Bar('Training epoch...', max=len(loader)) as bar:
            for src, tgt in loader:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = self.model.forward(src)
                    loss = self.loss_fn(pred.transpose(1, 2), tgt) # See loss_fn docs

                # See - https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.optimiser.zero_grad()

                bar.next()
        
        return loss.item()
        
    @torch.no_grad()
    def test_loss(self):
        """Calculates loss on held out test set.
        Returns:
            loss: loss on slice of test set.
        """
        self.model.eval()

        src, tgt = self.dataset.get_test(100)
        pred = self.model.forward(src)
        loss = self.loss_fn(pred.transpose(1, 2), tgt) 
        
        return loss.item()
