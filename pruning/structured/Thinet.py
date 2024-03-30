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
        # prune the model and return it.
        # Iterate through each layer of the model in reverse order 
        prev_module = None    
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                print(module)
                l1_norm = torch.norm(module.weight.data, p=1, dim=[1, 2, 3])  # Compute L1 norm of each filter and store it in a tensor
                print(l1_norm)
                print(module.out_channels)
                k = int(module.out_channels * self.amount)  # Prune k of the total channels
                # Find the least k  channels with the smallest L1 norm
                print(k)
                _, indices = torch.topk(l1_norm, k , largest=False)
                print(indices)
                # Create a mask that selects only the channels to keep
                mask = torch.ones(module.out_channels, dtype=torch.bool)
                mask[indices] = 0

                # Apply the mask to the weight and bias tensors
                module.weight.data = module.weight.data[mask]
                if module.bias is not None:
                    module.bias.data = module.bias.data[mask]
                    
                # Update the number of output channels
                module.out_channels = module.out_channels - k
                
                print(f"Pruned {name} layer")
                # If the next module is a Conv2d layer, update its number of input channels
                if prev_module is not None and isinstance(prev_module, torch.nn.Conv2d):
                    prev_module.in_channels = module.out_channels
                    prev_module.weight.data = prev_module.weight.data[:, mask, :, :]
                    
                if prev_module is not None and isinstance(prev_module, torch.nn.Linear) and isinstance(module, torch.nn.Conv2d):
                    # Calculate the size of the output feature map of the Conv2d layer
                    output_height = (module.input_height - module.kernel_size[0]) // module.stride[0] + 1
                    output_width = (module.input_width - module.kernel_size[1]) // module.stride[1] + 1

                    # Calculate the number of input features of the Linear layer
                    prev_module.in_features = module.out_channels * output_height * output_width

                    # Reshape the mask to match the number of input features of the Linear layer
                    reshaped_mask = mask.view(module.out_channels, output_height, output_width).all(dim=[1, 2])

                    # Apply the reshaped mask to the weight tensor of the Linear layer
                    prev_module.weight.data = prev_module.weight.data[:, reshaped_mask]
                    
            prev_module = module
        return self.model