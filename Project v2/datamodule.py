import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch
from  split_data import collect_data


class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset ,batch_size=64, num_workers=11, data_root="./data"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root

        self.train_data = train_dataset
        self.val_data = val_dataset
        self.test_data = test_dataset


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True, shuffle=False)
    
    # Data augmentation
    def get_transforms(self,split):
        mean = 0.5
        std = 0.2
        
        shared_transforms = [
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std) 
        ]
        
        if split == "train":
            return transforms.Compose([
                *shared_transforms,
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(4),
            ])
            
        elif split == "val":
            return transforms.Compose([
                *shared_transforms,
            ])
        elif split == "test":
            return transforms.Compose([
                *shared_transforms,
            ])