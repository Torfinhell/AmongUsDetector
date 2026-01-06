from .all_models import MyModel
import torch
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.fcos import FCOSClassificationHead
from src.losses import fcos_loss_fn
from functools import partial
from src.configs import ModelTrainConfig


class ModelFcosPretrained(MyModel):
    def __init__(
        self,
        cfg: ModelTrainConfig = ModelTrainConfig(),
        **kwargs,
    ):
        self.model_cfg = cfg.model_cfg
        super().__init__(cfg, **kwargs)

    def get_model(self):
        """
        Get FCOS model from torchvision configured for bounding box predictions.
        Returns a FCOS ResNet50 FPN model with pretrained backbone for custom number of classes.
        """
        model = fcos_resnet50_fpn(
            weights="DEFAULT",
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
        # for param in model.parameters():
        #     param.requires_grad = True
        return model

    def get_criterion(self):
        """
        Get the loss function for FCOS.
        Uses the built-in loss computation from FCOS model.
        Returns a wrapper function that extracts losses from model output.
        """
        return fcos_loss_fn

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for FCOS training.
        Uses SGD optimizer with momentum, which is standard for object detection.
        """
        # TODO Maybe other optimizer?
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.model_cfg.learning_rate,
            momentum=0.9,
            weight_decay=self.model_cfg.weight_decay,  # TODO
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.model_cfg.step_size,
            gamma=self.model_cfg.gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
