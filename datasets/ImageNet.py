import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

class ImageNet:
    def __init__(self, batch_size=128, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def get_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Assuming 'train' and 'val' folders in the path
        train_dir = os.path.join('./data', 'train')
        val_dir = os.path.join('./data', 'val')
        # Loading ImageNet dataset
        trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
        testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
        # Setting train data loader and test data loader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader
