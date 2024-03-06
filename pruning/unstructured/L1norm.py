#basic implementation of network based on L1 norm structured .
import os
import torch
import torch.nn.utils.prune as prune
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm

#trainer class for all models
import torch
import matplotlib.pyplot as plt
import math
from tqdm.auto import tqdm



#has to saperate next
class Trainer:
    def __init__(
        self,
        model,
        epochs,
        trainloader,
        criterion,
        optimizer,
     
    ):
        self.model = model
        self.epochs = epochs
        self.trainloader = trainloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": []}
        self.acc=[]
    def train_step(self):
        self.model.train() #setting model to training mode
        train_loss = 0

        #setting requires_grad =True
        with torch.set_grad_enabled(True):
            for batch, (X, y) in enumerate(self.trainloader):
                #forward pass
                y_pred = self.model(X)
                #calculate and sum loss
                loss = self.criterion(y_pred, y)
                train_loss += loss.item()
                #backward pass and setting previous epoch gradients to zero
                self.optimizer.zero_grad()
                loss.backward()
                #parameters update
                self.optimizer.step()
                #self.scheduler.step()
        return train_loss
    
        
    def train(self):
        #training 
        for epoch in tqdm(range(self.epochs)):
            train_loss = self.train_step()
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {math.log(train_loss):.4f} | "
            )
            self.loss["train"].append(math.log(train_loss)) #using log loss 
            
        return self.loss



class UnstructuredL1normPrune:
    def __init__(self, model, epochs,train_loader,  criterion,optimizer,pruning_rate =0.5):
        self.model = model
        self.pruning_rate = pruning_rate
        self.optimizer=optimizer
        self.epochs=epochs
        self.train_loader=train_loader
        self.criterion=criterion

    def prune_model(self):
        #prune the model and return it.
        self.model.eval()
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=self.pruning_rate)
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.pruning_rate)
        return self.model
    
    def train_prune_retrain(self):
        #train the model, prune it and retrain it.
        trainer = Trainer(self.model, self.epochs,self.train_loader, self.criterion, self.optimizer)
        trainer.train()
        print("Training is done")
        unstructured_prune = UnstructuredL1normPrune(self.model, self.epochs,self.train_loader, self.criterion, self.optimizer,self.pruning_rate)
        pruned_model = unstructured_prune.prune_model()
        print("Pruning is done")
        trainer = Trainer(pruned_model, self.epochs,self.train_loader, self.criterion, self.optimizer)
        trainer.train()
        print("Retraining after pruning is done")
        return pruned_model
        


