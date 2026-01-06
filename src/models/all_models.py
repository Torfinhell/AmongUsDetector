import torch
import lightning as L
from src.metrics import PRauc, nms
from src.data_module import generate
from src.configs import ModelTrainConfig


class MyModel(L.LightningModule):
    def __init__(self, cfg: ModelTrainConfig, augment=False):
        super().__init__()
        self.model = self.get_model()
        self.criterion = self.get_criterion()
        self.augment = augment
        self.train_losses = []
        self.val_losses = []
        self.metrics_cfg = cfg.metric_cfg
        self.create_cfg = cfg.creation_cfg
        self.mAP_score = PRauc(self.metrics_cfg)
        self.use_nms = cfg.model_cfg.use_nms
        self.nms_thr = cfg.model_cfg.nms_thr
        if self.use_nms:
            self.mAP_score_nms = PRauc(self.metrics_cfg)

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
                f"Epoch {self.current_epoch} - Train Loss: {self.train_losses[-1]:.6f}"
            )

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch"""
        avg_loss = self.trainer.callback_metrics.get("loss_val_epoch", None)
        if avg_loss is not None:
            self.val_losses.append(
                avg_loss.item() if hasattr(avg_loss, "item") else float(avg_loss)
            )
            print(f"Epoch {self.current_epoch} - Val Loss: {self.val_losses[-1]:.6f}")
        mAP_score = self.mAP_score.compute()
        self.log_dict(
            {
                "mAP_score": mAP_score,
            }
        )
        print(f"Epoch {self.current_epoch} - mAP: {mAP_score}")
        self.mAP_score.reset()
        if self.use_nms:
            mAP_score_nms = self.mAP_score_nms.compute()
            self.log_dict(
                {
                    "mAP_score_nms": mAP_score_nms,
                }
            )
            print(f"Epoch {self.current_epoch} - mAP_nms: {mAP_score_nms}")
            self.mAP_score_nms.reset()
        if self.create_cfg.generate_every_epoch:
            generate(self.create_cfg)

    def _step(self, batch, kind):
        images, targets = batch
        self.model.train()
        if kind == "train":
            preds = self.model(images, targets)
            loss = self.criterion(preds, targets)
        else:
            with torch.no_grad():
                preds = self.model(images, targets)
            loss = self.criterion(preds, targets)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(images, targets)
            self.mAP_score.update(preds, targets)
            if self.use_nms:
                preds = nms(preds, iou_thr=self.nms_thr)
                self.mAP_score_nms.update(preds, targets)
        self.log(f"loss_{kind}", loss.item(), on_step=True, on_epoch=True)
        return loss
