import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes
from tops.config import LazyCall as L
from SSD.ssd.data.lidar_loader import LidarDetectionDataset
from ssd import utils
from ssd.data.transforms import Normalize, ToTensor, GroundTruthBoxesToAnchors
from .utils import get_dataset_dir, get_output_dir

# The line belows inherits the configuration set for the SSD300 lidar trondheim config
from .ssd300 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    # train_cpu_transform,
    # val_cpu_transform,
    # gpu_transform,
    label_map,
    anchors
)
# We can keep all other configs the same, and only change the learning rate to a given value.
# You can now start training with the following command: python train.py configs/change_lr.py

train.batch_size = 128
optimizer = L(torch.optim.Adam)(
    # Tip: Scale the learning rate by batch size! 2.6e-3 is set for a batch size of 32. use 2*2.6e-3 if you use 64
    lr=train.batch_size/32*2e-4, weight_decay=5e-4
)
backbone = L(backbones.NormalizedModel)(
    output_channels=[128, 256, 128, 128, 64, 64],
    image_channels=3,
    output_feature_sizes=[[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]],
)

model.feature_extractor = backbone