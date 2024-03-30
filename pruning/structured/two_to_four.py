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


class two_to_four_prune:
    def __init__(self, model, epochs, train_loader, criterion, optimizer):
        self.model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.criterion = criterion
       

    def prune_model(self):
        # prune the model and return it.
        # Iterate through each layer of the model
        model = copy.deepcopy(self.model)
        for name, module in model.named_children():
            # Apply pruning to the weight parameter of the module
            inst =PruningMethod(module)
            inst.two_to_four_structured(module, 'weight', module)
            prune.remove(module, 'weight')
        return model

    def train_prune_retrain(self):
        # train the model, prune it and retrain it.
        trainer = Trainer(self.model, self.epochs, self.train_loader, self.criterion, self.optimizer)
        trainer.train()
        model = copy.deepcopy(self.model)
        print("Training is done")
        unstructured_prune = two_to_four_prune(self.model, self.epochs, self.train_loader, self.criterion,
                                                     self.optimizer)
        pruned_model = unstructured_prune.prune_model()
        print("Pruning is done")
        # trainer = Trainer(pruned_model, self.epochs, self.train_loader, self.criterion, self.optimizer)
        # trainer.train()
        # print("Retraining after pruning is done")
        return model, pruned_model
    

#class to prune one moduleof model (like conv2d or linear)
class PruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    def __init__(self,module):
        self.weights=module.weight
        

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        shape = self.weights.shape
        if len(shape) == 4:  # Convolutional layer
            for i in range(shape[0]): 
                for j in range(shape[1]):  
                    for k in range(shape[2]): 
                        for l in range(0,shape[3]-3,4):  # Iterate over the fourth dimension up to the third-to-last element
                            # Find the 1 and 2 minimum value in the current group of 4 contiguous elements
                            sorted_vals = torch.sort(torch.abs(self.weights[i, j, k, l:l+4])).values
                            min_val, second_min_val = sorted_vals[:2]
                            # Update the mask where the minimum value is found
                            mask[i, j, k, l:l+4] = torch.where((torch.abs(self.weights[i, j, k, l:l+4]) == min_val) | (torch.abs(self.weights[i, j, k, l:l+4]) == second_min_val), 
                                                                 torch.zeros(4),  
                                                                 torch.ones(4)) 
        elif len(shape) == 2:  # Linear layer
            for i in range(shape[0]):
                for j in range(0, shape[1] - 3, 4):  # Iterate over the second dimension
                    sorted_vals = torch.sort(self.weights[i, j:j+4]).values
                    min_val, second_min_val = sorted_vals[:2]
                    # Update the mask where the minimum value is found
                    mask[i, j:j+4] = torch.where((self.weights[i, j:j+4] == min_val) | 
                                                 (self.weights[i, j:j+4] == second_min_val), 
                                                 torch.zeros(4),  
                                                 torch.ones(4))

            
        return mask
    
    def two_to_four_structured(self,module, name,modules):
        PruningMethod.apply(module, name,modules)
        return module
