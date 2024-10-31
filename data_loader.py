# data_loader.py
import torch
import torchvision
import torchvision.transforms as transforms

# ======================================================================================================
# download mnist or cifar10 as training and testing datasets (60000 or 50000 + 10000)
import torchvision
import torchvision.transforms as transforms
import torch

def load_dataset(DataSet_Name, BatchSize):
    if DataSet_Name == 'CIFAR10':
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transf)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize, shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transf)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BatchSize, shuffle=False, num_workers=4)
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return trainset, trainloader, testset, testloader, classes

    elif DataSet_Name == 'MNIST':
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transf)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize, shuffle=True, num_workers=4)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transf)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BatchSize, shuffle=False, num_workers=4)
        classes = tuple(str(i) for i in range(10))  # MNIST classes 0-9
        return trainset, trainloader, testset, testloader, classes

    elif DataSet_Name == 'Fashion-MNIST':
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # Normalization values for Fashion-MNIST
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transf)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize, shuffle=True, num_workers=4)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transf)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BatchSize, shuffle=False, num_workers=4)
        classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
        return trainset, trainloader, testset, testloader, classes

    else:
        raise ValueError("Unsupported dataset. Please use 'CIFAR10', 'MNIST', or 'Fashion-MNIST'.")
