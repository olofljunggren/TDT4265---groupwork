import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch
from  split_data import collect_data




class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=11, data_root="./data", train_split_ratio=0.8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.train_split_ratio = train_split_ratio

        # Collect data
        collect_data()


    def prepare_data(self):
        # Download the dataset if needed (only using rank 1)
        data_dir = "data/train/images"
        datasets.ImageFolder(root=data_dir, transform=self.get_transforms("train"))

        # TODO instead use our own data
    

       
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        test_dataset = datasets.CocoCaptions(root=self.data_root, train=False, transform=self.get_transforms("test"))
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True, shuffle=False)
    
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