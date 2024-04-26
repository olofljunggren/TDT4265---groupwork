import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.label_file = label_file
        self.transform = transform

        # Read labels from the label file
        with open(label_file, 'r') as f:
            self.labels = np.loadtxt(f, dtype=np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load image
        idx_fill = "frame_"+str(idx).zfill(6)
        img_name = os.path.join(self.image_dir, f"{idx_fill}.PNG")
        image = Image.open(img_name)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.labels[idx]

        return image, label
