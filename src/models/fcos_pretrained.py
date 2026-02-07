from .all_models import MyModel
import torch
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.fcos import FCOSClassificationHead
from src.losses import fcos_loss_fn
from functools import partial
from src.configs import ModelTrainConfig
from functools import partial


class ModelFcosPretrained(MyModel):
    def __init__(
        self,
        cfg: ModelTrainConfig
    ):
        self.model_cfg = cfg.model_cfg
        if(getattr(cfg, "training_cfg", None) is not None):
            self.num_epochs = cfg.training_cfg.num_epochs
        super().__init__(cfg)

    def get_model(self):
        """
        Get FCOS model from torchvision configured for bounding box predictions.
        Returns a FCOS ResNet50 FPN model with pretrained backbone for custom number of classes.
        """
        model = fcos_resnet50_fpn(
            weights="DEFAULT" if self.model_cfg.is_pretrained else None,
            nms_thresh=self.model_cfg.nms_thr,
            score_thresh=self.model_cfg.score_thresh,
            trainable_backbone_layers=self.model_cfg.backbone_layers,
            image_mean=None,
            image_std=None,
            detections_per_img=self.model_cfg.detections_per_img,
            topk_candidates=self.model_cfg.topk_candidates,
        )
        model.head.classification_head = FCOSClassificationHead(
            in_channels=self.model_cfg.head_in_channels,
            num_anchors=self.model_cfg.num_anchors
            or model.head.classification_head.num_anchors,
            num_classes=self.model_cfg.num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )
        model.transform.min_size = (self.model_cfg.min_size,)
        model.transform.max_size = self.model_cfg.max_size
        return model

    def get_criterion(self):
        """
        Get the loss function for FCOS.
        Uses the built-in loss computation from FCOS model.
        Returns a wrapper function that extracts losses from model output.
        """
        return partial(fcos_loss_fn, lambdas=self.model_cfg.lambdas)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for FCOS training.
        Uses SGD optimizer with momentum, which is standard for object detection.
        """
        optimizer = torch.optim.SGD(
            [
                {
                    "params": self.model.backbone.parameters(),
                    "lr": self.model_cfg.backbone_learning_rate,
                },
                {
                    "params": self.model.head.parameters(),
                    "lr": self.model_cfg.heads_learning_rate,
                },
            ],
            momentum=0.9,
            weight_decay=self.model_cfg.weight_decay,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }