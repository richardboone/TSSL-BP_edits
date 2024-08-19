import os
import torchvision.datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split




def get_mnist(data_path, network_config):
   print("loading MNIST")
   if not os.path.exists(data_path):
       os.mkdir(data_path)


   batch_size = network_config['batch_size']
   # val_size = network_config['val_size']


   transform_train = transforms.Compose([
       # transforms.RandomCrop(28, padding=4),
       # transforms.RandomHorizontalFlip(),
       # transforms.RandomRotation(30),
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ])


   transform_test = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ])
   trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
   testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)


   # train_indices, val_indices, _, _ = train_test_split(
   # range(len(trainset)),
   # trainset.targets,
   # stratify=trainset.targets,
   # test_size=val_size,)


   # train_split = Subset(trainset, train_indices)
   # val_split = Subset(trainset, val_indices)


   trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
   # valloader = torch.utils.data.DataLoader(val_split, batch_size=batch_size, shuffle=True, num_workers=4)
   testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
   return trainloader, testloader
