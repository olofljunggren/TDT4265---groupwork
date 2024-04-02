from torch import nn
from ssd.modeling.backbones.vgg import VGG
from ssd.modeling.backbones.basic import BasicModel
from ssd.modeling.backbones.batch_normalization import NormalizedModel
from tops.config import instantiate
from ssd.utils import load_config
from ssd import torch_utils

class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        print(
            "Detector initialized. Total Number of params: ",
            f"{torch_utils.format_params(self)}")
        print(
            f"Backbone number of parameters: {torch_utils.format_params(self.backbone)}")
        print(
            f"SSD Head number of parameters: {torch_utils.format_params(self.box_head)}")
        
    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections
    
def build_backbone(cfg):
    backbone_name = cfg.backbone.name
    if backbone_name == "basic":
        cfg = load_config("../configs/ssd300.py")
    if backbone_name == "batch_normalization":
        cfg = load_config("../configs/change_lr.py")
    # if backbone_name == "vgg":
    #     model = VGG(cfg)
    #     if cfg.MODEL.BACKBONE.PRETRAINED:
    #         state_dict = load_state_dict_from_url(
    #             "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth")
    #         model.init_from_pretrain(state_dict)
    model = instantiate(cfg.model)
    return model