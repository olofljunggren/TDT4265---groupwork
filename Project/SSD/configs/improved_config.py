import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes
from tops.config import LazyCall as L
from ssd.data.mnist import MNISTDetectionDataset
from ssd import utils
from ssd.data.transforms import Normalize, ToTensor, GroundTruthBoxesToAnchors, RandomSampleCrop, RandomHorizontalFlip, Resize
from .utils import get_dataset_dir, get_output_dir

# The line belows inherits the configuration set for the SSD300 mnist config
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
train.epochs = 32
train.batch_size = 128
optimizer = L(torch.optim.AdamW)(
    # Tip: Scale the learning rate by batch size! 2.6e-3 is set for a batch size of 32. use 2*2.6e-3 if you use 64
    lr=train.batch_size/32*2e-4, weight_decay=1e-5
)
backbone = L(backbones.Improved)(
    output_channels=[128, 256, 128, 128, 64, 64],
    image_channels=3,
    output_feature_sizes=[[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]],
)

model.feature_extractor = backbone

data_train = dict(
    dataset=L(MNISTDetectionDataset)(
        data_dir=get_dataset_dir("mnist_object_detection/train"),
        is_train=True,
        transform=L(torchvision.transforms.Compose)(transforms=[
            L(ToTensor)(),  # ToTensor has to be applied before conversion to anchors.
            # GroundTruthBoxesToAnchors assigns each ground truth to anchors, required to compute loss in training.
            L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
        ])
    ),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}", num_workers=4, pin_memory=True, shuffle=True, batch_size="${...train.batch_size}", collate_fn=utils.batch_collate,
        drop_last=True
    ),
    # GPU transforms can heavily speedup data augmentations.
    gpu_transform=L(torchvision.transforms.Compose)(transforms=[
        # L(Resize)(imshape=(38,38)),
        # L(RandomSampleCrop)(),
        # L(RandomHorizontalFlip)(p=0.5),
        L(Normalize)(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize has to be applied after ToTensor (GPU transform is always after CPU)
    ])
)