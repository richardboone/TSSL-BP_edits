import os
import torchvision.datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import torch
from tqdm import tqdm


def pad_seq(batch):
   # Let's assume that each element in "batch" is a tuple (data, label).
   # Sort the batch in the descending order
   sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
   # Get each sequence and pad it
   sequences = [torch.as_tensor(x[0]) for x in sorted_batch]
   sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
   # Also need to store the length of each sequence
   # This is later needed in order to unpad the sequences
   lengths = torch.as_tensor([x.shape[0] for x in sequences]) # lengths = torch.LongTensor([len(x) for x in sequences])
   # Don't forget to grab the labels of the *sorted* batch
   labels = torch.as_tensor([x[1] for x in sorted_batch])  # labels = torch.LongTensor(map(lambda x: x[1], sorted_batch))
   return sequences_padded, lengths, labels


def get_dvsg(data_path, network_config):
   print("loading DVSG")
   n_steps = network_config['n_steps']


   if not os.path.exists(data_path):
       os.mkdir(data_path)
      
   batch_size = network_config['batch_size']
   val_size = network_config['val_size']


   print("Loading data - Example mode 2")
   trainset = DVS128Gesture(root=data_path, train=True, data_type='frame', split_by='time', duration=300000, frames_number=n_steps)
   testset = DVS128Gesture(root=data_path, train=False, data_type='frame', split_by='time', duration=300000,  frames_number=n_steps)
   print(f'dataset_train:{trainset.__len__()}, dataset_test:{testset.__len__()}')


   # train_indices, val_indices, _, _ = train_test_split(
   # range(len(trainset)),
   # trainset.targets,
   # stratify=trainset.targets,
   # test_size=val_size,)


   # train_split = Subset(trainset, train_indices)
   # val_split = Subset(trainset, val_indices)


   print("Creating data loaders")
   # Collate function is needed because each sample may have a different size
   trainloader = torch.utils.data.DataLoader(
       trainset, batch_size=batch_size, collate_fn=pad_seq,
       shuffle=True, num_workers=4, pin_memory=False)
   # valloader = torch.utils.data.DataLoader(
   #     val_split, batch_size=batch_size, collate_fn=pad_seq,
   #     shuffle=True, num_workers=4, pin_memory=False)
   testloader = torch.utils.data.DataLoader(
       testset, batch_size=batch_size, collate_fn=pad_seq,
       shuffle=False, num_workers=4, pin_memory=False)


   return trainloader, testloader