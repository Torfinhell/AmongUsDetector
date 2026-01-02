from .all_models import MyModel
import torch
from torchvision.models.detection import fcos_resnet50_fpn
from src.losses import fcos_loss_fn
from dataclasses import dataclass


@dataclass
class ModelFcosPretrainedConfig:
    num_classes: int = 18
    learning_rate: float = 4e-3
    step_size: int = 3
    weight_decay: float = 0.0005
    gamma: float = 0.98


class ModelFcosPretrained(MyModel):
    def __init__(
        self,
        training_cfg: ModelFcosPretrainedConfig = ModelFcosPretrainedConfig(),
        **kwargs,
    ):
        self.training_cfg = training_cfg
        super().__init__(**kwargs)

    def get_model(self):
        """
        Get FCOS model from torchvision configured for bounding box predictions.
        Returns a FCOS ResNet50 FPN model with pretrained backbone for custom number of classes.
        """
        # Load model without pretrained weights to allow custom num_classes
        model = fcos_resnet50_fpn(
            pretrained=False,
            num_classes=self.training_cfg.num_classes,
            pretrained_backbone=True,
            trainable_backbone_layers=5,
        )
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
            lr=self.training_cfg.learning_rate,
            momentum=0.9,
            weight_decay=self.training_cfg.weight_decay,  # TODO
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.training_cfg.step_size,
            gamma=self.training_cfg.gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
