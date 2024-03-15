import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class CIFAR10:
    def __init__(self, batch_size=128, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def get_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        # Downloading train and test sets
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        # Setting train data loader and test data loader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader
