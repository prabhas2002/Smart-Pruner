from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())
import torch
import matplotlib.pyplot as plt
import math
from tqdm.auto import tqdm

# has to saperate next
class Trainer:
    def __init__(
            self,
            model,
            epochs,
            train_loader,
            criterion,
            optimizer,

    ):
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": []}
        self.acc = []

    def train_step(self):
        self.model.train()  # setting model to training mode
        train_loss = 0

        # setting requires_grad =True
        with torch.set_grad_enabled(True):
            for batch, (X, y) in enumerate(self.train_loader):
                # forward pass
                y_pred = self.model(X)
                # calculate and sum loss
                loss = self.criterion(y_pred, y)
                train_loss += loss.item()
                # backward pass and setting previous epoch gradients to zero
                self.optimizer.zero_grad()
                loss.backward()
                # parameters update
                self.optimizer.step()
                # self.scheduler.step()
        return train_loss

    def train(self):
        # training
        for epoch in tqdm(range(self.epochs)):
            train_loss = self.train_step()
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {math.log(train_loss):.4f} | "
            )
            self.loss["train"].append(math.log(train_loss))  # using log loss

        return self.loss

