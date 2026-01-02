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
        if kind == "train":
            self.model.train()
            preds = self.model(images)
            loss = self.criterion(preds, targets)
        else:
            self.model.eval()
            with torch.no_grad():
                preds = self.model(images)
            loss = self.criterion(preds, targets)

        self.log(f"loss_{kind}", loss.item(), on_step=True, on_epoch=True)
        return loss
