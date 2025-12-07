from .all_models import MyModel
import torch
from torchvision.models.detection import fcos_resnet50_fpn


class ModelFcos(MyModel):
    def __init__(self, num_classes=2, **kwargs):
        self.num_classes = num_classes
        super().__init__(**kwargs)

    def get_model(self):
        """
        Get FCOS model from torchvision configured for bounding box predictions.
        Returns a FCOS ResNet50 FPN model with pretrained backbone for custom number of classes.
        """
        # Load model without pretrained weights to allow custom num_classes
        model = fcos_resnet50_fpn(
            pretrained=False,
            num_classes=self.num_classes,
            pretrained_backbone=True,
            trainable_backbone_layers=3,
        )

        # FCOS is natively designed for bounding box prediction with:
        # - Center-ness prediction for each location
        # - Bounding box regression (left, top, right, bottom)
        # - Class prediction for each location
        return model

    def get_criterion(self):
        """
        Get the loss function for FCOS.
        Uses the built-in loss computation from FCOS model.
        Returns a wrapper function that extracts losses from model output.
        """

        def fcos_loss_fn(predictions, targets):
            """
            FCOS model returns a dictionary with losses when in training mode.
            This function extracts and sums the total loss.
            """
            if isinstance(predictions, dict) and "loss_classifier" in predictions:
                # Model is in training mode and returns losses
                total_loss = sum(
                    loss
                    for loss in predictions.values()
                    if isinstance(loss, torch.Tensor)
                )
                return total_loss
            else:
                # Fallback: use smooth L1 loss if needed
                raise RuntimeError(
                    "FCOS model should return loss dict in training mode"
                )

        return fcos_loss_fn

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for FCOS training.
        Uses SGD optimizer with momentum, which is standard for object detection.
        """
        # SGD optimizer with momentum is standard for FCOS
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.02,  # Initial learning rate (typical for FCOS)
            momentum=0.9,  # Momentum for better convergence
            weight_decay=0.0005,  # L2 regularization
        )

        # Learning rate scheduler to decay learning rate over time
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,  # Decay LR every 3 epochs
            gamma=0.1,  # Multiply LR by 0.1 every step_size epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Update LR every epoch
                "frequency": 1,
            },
        }
