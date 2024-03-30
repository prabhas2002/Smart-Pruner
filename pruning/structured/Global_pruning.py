# basic implementation of network based on L1 norm structured .
import copy
from pathlib import Path

print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())
import os
import torch
import torch.nn.utils.prune as prune
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm

from pruning.Train import Trainer


class GlobalPrune:
    def __init__(self, model=None, epochs=None, train_loader=None, criterion=None, optimizer=None, pruning_rate=0.5):
        self.model = model
        self.pruning_rate = pruning_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.criterion = criterion

    def setargs(self, model, epochs, train_loader, criterion, optimizer, pruning_rate=0.5):
        self.model = model
        self.pruning_rate = pruning_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.criterion = criterion

    def prune_model(self):
        # prune the model and return it.
        # Specify the parameters to prune
        model = copy.deepcopy(self.model)
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                parameters_to_prune.append((module, 'weight'))

        # Convert the list to a tuple
        parameters_to_prune = tuple(parameters_to_prune)

        # Prune the parameters
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured, #can change this to any other also.
            amount= self.pruning_rate,
        )
        return model

    def train_prune_retrain(self):
        # train the model, prune it and retrain it.
        trainer = Trainer(self.model, self.epochs, self.train_loader, self.criterion, self.optimizer)
        trainer.train()
        model = copy.deepcopy(self.model)
        print("Training is done")
        unstructured_prune = GlobalPrune(self.model, self.epochs, self.train_loader, self.criterion,
                                                     self.optimizer, self.pruning_rate)
        pruned_model = unstructured_prune.prune_model()
        print("Pruning is done")
        # trainer = Trainer(pruned_model, self.epochs, self.train_loader, self.criterion, self.optimizer)
        # trainer.train()
        # print("Retraining after pruning is done")
        return model, pruned_model
