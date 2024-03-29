import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class CIFAR100:
    def __init__(self, batch_size=128, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def get_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        # Downloading train and test sets
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        # Setting train data loader and test data loader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader
