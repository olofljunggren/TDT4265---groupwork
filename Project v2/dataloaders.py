from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import v2
import torch
import typing
import numpy as np
import pathlib
np.random.seed(0)
from  split_data import collect_data

# Collect data
collect_data()

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)


def get_data_dir():
    server_dir = pathlib.Path("/datasets/tdt4265/ad/NAPLab-LiDAR")
    if server_dir.is_dir():
        return str(server_dir)
    return "data"


def load_dataset(batch_size: int, validation_fraction: float = 0.1
                 ) -> typing.List[torch.utils.data.DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!

    # Used for optimal init in task 4
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), # Randomly flips the image horizontally
        transforms.RandomCrop(32, padding=4), # Pads the image and randomly crops it back to its original size
        transforms.RandomRotation(4),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    data_train = datasets.CIFAR10(get_data_dir(),
                                  train=True,
                                  download=True,
                                  transform=transform_train)

    data_test = datasets.CIFAR10(get_data_dir(),
                                 train=False,
                                 download=True,
                                 transform=transform_test)
    

    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_val, dataloader_test



load_dataset()