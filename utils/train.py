#have to write singlr training class for all models
import torch
import math
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