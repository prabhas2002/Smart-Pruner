# basic implementation of network based on L1 norm structured .
from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())
import os
import torch
import torch.nn.utils.prune as prune
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm

from train import Trainer

class UnstructuredL1normPrune:
    def __init__(self, model, epochs, train_loader, criterion, optimizer, pruning_rate=0.5):
        self.model = model
        self.pruning_rate = pruning_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.criterion = criterion

    def prune_model(self):
        # prune the model and return it.
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=self.pruning_rate)
                prune.remove(module, name='weight')
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.pruning_rate)
                prune.remove(module, name='weight')
        return self.model

    def train_prune_retrain(self):
        # train the model, prune it and retrain it.
        trainer = Trainer(self.model, self.epochs, self.train_loader, self.criterion, self.optimizer)
        trainer.train()
        print("Training is done")
        unstructured_prune = UnstructuredL1normPrune(self.model, self.epochs, self.train_loader, self.criterion,
                                                     self.optimizer, self.pruning_rate)
        pruned_model = unstructured_prune.prune_model()
        print("Pruning is done")
        trainer = Trainer(pruned_model, self.epochs, self.train_loader, self.criterion, self.optimizer)
        trainer.train()
        print("Retraining after pruning is done")
        return pruned_model
