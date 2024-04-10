from datamodule import CocoCaptionsDataModule
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
    

        self.loss_fn = IoU()
        self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.max_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

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
    
    dm = CocoCaptionsDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_split_ratio=config.train_split_ratio,
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
