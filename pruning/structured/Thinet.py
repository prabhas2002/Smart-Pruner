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

class Thinet:
    def __init__(self, model, epochs, train_loader, criterion, optimizer, amount=0.5):
        self.model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.criterion = criterion
        self.amount = amount
    def prune_model(self):
        model = copy.deepcopy(self.model)
        prev_module = None    
        for name, module in list(model.named_modules()):
            if isinstance(module, torch.nn.Conv2d):
                l1_norm = torch.norm(module.weight.data, p=1, dim=[1, 2, 3])
                k = int(module.out_channels * self.amount)
                _, indices = torch.topk(l1_norm, k , largest=False)
                mask = torch.ones(module.out_channels, dtype=torch.bool)
                mask[indices] = 0

                module.weight.data = module.weight.data[mask, :, :, :]
                if module.bias is not None:
                    module.bias.data = module.bias.data[mask]
                    
                module.out_channels = module.out_channels - k
                
                if prev_module is not None and isinstance(prev_module, torch.nn.Conv2d):
                    module.in_channels = prev_module.out_channels
                    
            prev_module = module
           
        return model