import torch
import lightning as L


class MyModel(L.LightningModule):
    def __init__(self, augment=False):
        super().__init__()
        self.model = self.get_model()
        self.criterion = self.get_criterion()
        self.augment = augment
        self.train_losses = []
        self.val_losses = []

    def get_model(self):
        raise NotImplementedError(
            "Should not use base class. Should implement get_model for child class"
        )

    def get_criterion(self):
        raise NotImplementedError(
            "Should not use base class. Should implement get_criterion for child class"
        )

    def configure_optimizers(self):
        raise NotImplementedError(
            "Should not use base class. Should implement configure_optimizers for child class"
        )

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        avg_loss = self.trainer.callback_metrics.get("loss_train_epoch", None)
        if avg_loss is not None:
            self.train_losses.append(
                avg_loss.item() if hasattr(avg_loss, "item") else float(avg_loss)
            )
            print(
                f"\nEpoch {self.current_epoch} - Train Loss: {self.train_losses[-1]:.6f}"
            )

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch"""
        avg_loss = self.trainer.callback_metrics.get("loss_val_epoch", None)
        if avg_loss is not None:
            self.val_losses.append(
                avg_loss.item() if hasattr(avg_loss, "item") else float(avg_loss)
            )
            print(f"Epoch {self.current_epoch} - Val Loss: {self.val_losses[-1]:.6f}")

    def _step(self, batch, kind):
        images, targets = batch

        # Get device - use self.device which Lightning sets automatically
        device = self.device

        # Move images to device (they come as list from collate_fn)
        images = [img.to(device) for img in images]

        # Move targets to device
        targets_list = []
        for target in targets:
            target_dict = {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
            }
            targets_list.append(target_dict)

        # Set model mode based on training/validation
        if kind == "train":
            self.model.train()
            # FCOS model in training mode returns losses dict
            loss_dict = self.model(images, targets_list)
            # Sum key losses (bbox_regression and bbox_ctrness are the main FCOS losses)
            loss = loss_dict["bbox_regression"] + loss_dict["bbox_ctrness"]
        else:
            # In validation mode, also compute losses
            self.model.train()  # Keep in train mode to get losses
            with torch.no_grad():
                loss_dict = self.model(images, targets_list)
                # Sum key losses
                loss = loss_dict["bbox_regression"] + loss_dict["bbox_ctrness"]

        self.log(f"loss_{kind}", loss.item(), on_step=True, on_epoch=True)
        return loss
