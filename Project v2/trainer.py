from datamodule import DataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import nn
import torchvision.ops as ops
from torchvision.models import resnet50, ResNet50_Weights, detection
from torchmetrics import Accuracy
import torchvision.models.detection as detection
import munch
import yaml
from pathlib import Path
from dataset import CustomDataset
from split_data import collect_data, create_labels
from torchvision.models.detection.rpn import AnchorGenerator


torch.set_float32_matmul_precision('medium')
config = munch.munchify(yaml.load(open("config.yaml"), Loader=yaml.FullLoader))

# IoU loss functions
# Do we need a class to each loss function as well???
def GIoU_loss(self, predicted_boxes, target_boxes):
    return ops.generalized_box_iou_loss(predicted_boxes, target_boxes)

def DIoU_loss(predicted_boxes, target_boxes):
    return ops.distance_box_iou_loss(predicted_boxes, target_boxes)

def CIoU_loss(boxes1, boxes2, reduction='none', eps=1e-7):
    #eps: small number to prevent division by zero. The default value is 1e-7
    return ops.complete_box_iou_loss(boxes1, boxes2, reduction=reduction, eps=eps)

# loss funjctions up a head^

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # Freeze all the parameters in the backbone network
        for param in model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer

        # Unfreeze the parameters in the last CNN layer (the backbone network)
        for param in model.backbone[-1].parameters():
            param.requires_grad = True

        # Define your custom classifier layer using torch.nn.Sequential
        classifier_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=512),  # Example linear layer
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=self.config.num_classes)  # Adjust num_classes as per your dataset
        )

        # Replace the classifier layer in the model
        model.roi_heads.box_predictor.cls_score = classifier_layer
        # self.model.fc = nn.Linear(2048, self.config.num_classes)
    

        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)
    
    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.config.max_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.max_lr, weight_decay=self.config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat
    
    # Define the model architecture
    def get_model(num_classes):
        # Load a pre-trained backbone network
        backbone = resnet50(pretrained=True).features

        # Define the number of input channels for the backbone
        backbone.out_channels = 1280

        # Anchor generator
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

        # ROI Pooler
        roi_pooler = ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        # Define the model
        model = detection.FasterRCNN(backbone,
                        num_classes=num_classes,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler)

        return model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.acc_fn(y_hat, y)
        self.log_dict({
            "train/loss": loss,
            "train/acc": acc
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        acc = self.acc_fn(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.log_dict({
            "val/loss":loss,
            "val/acc": acc
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        acc = self.acc_fn(y_hat, y)
        self.log_dict({
            "test/acc": acc,
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

if __name__ == "__main__":
    
    pl.seed_everything(42)
    # Collect data
    collect_data(config.train_split_ratio)
    create_labels("data/train/")
    create_labels("data/validation/")
    create_labels("data/test/")


    data_dir_train = "data/train/images"
    data_dir_val = "data/validation/images"
    data_dir_test = "data/test/images"
    
    labels_train = "data/train/labels.txt"
    labels_val = "data/validation/labels.txt"
    labels_test = "data/test/labels.txt"

    train_dataset = CustomDataset(image_dir=data_dir_train, label_file=labels_train, transform=DataModule.get_transforms("train"))
    val_dataset = CustomDataset(image_dir=data_dir_val, label_file=labels_val, transform=DataModule.get_transforms("val"))
    test_dataset = CustomDataset(image_dir=data_dir_test, label_file=labels_test, transform=DataModule.get_transforms("test"))
    
    dm = DataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        data_root=config.data_root
    )
    if config.checkpoint_path:
        model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        print("Loading weights from checkpoint...")
    else:
        model = LitModel(config)

    trainer = pl.Trainer(
        devices=config.devices, 
        max_epochs=config.max_epochs, 
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_progress_bar=config.enable_progress_bar,
        precision="bf16-mixed",
        # deterministic=True,
        logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
        callbacks=[
            EarlyStopping(monitor="val/acc", patience=config.early_stopping_patience, mode="max", verbose=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
                            filename='best_model:epoch={epoch:02d}-val_acc={val/acc:.4f}',
                            auto_insert_metric_name=False,
                            save_weights_only=True,
                            save_top_k=1),
        ])
    if not config.test_model:
        trainer.fit(model, datamodule=dm)
    
    trainer.test(model, datamodule=dm)
