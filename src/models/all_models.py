import torch
import lightning as L
from src.metrics import PRauc
from src.configs import ModelTrainConfig
from src.utils import nms, calculate_norm_grad, drawbboxes

class MyModel(L.LightningModule):
    def __init__(self, cfg: ModelTrainConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.get_model()
        self.criterion = self.get_criterion()
        self.train_losses = []
        self.val_losses = []
        self.metrics_cfg = cfg.metric_cfg
        self.mAP_score = {"test":PRauc(self.metrics_cfg), "val":PRauc(self.metrics_cfg)}
        self.use_nms = cfg.model_cfg.use_nms
        self.nms_thr = cfg.model_cfg.nms_thr
        self.batch_val_show=cfg.datamodule_cfg.batch_val_show
        if self.use_nms:
            self.mAP_score_nms = {"test":PRauc(self.metrics_cfg), "val":PRauc(self.metrics_cfg)}
        

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
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val", batch_idx)
    def predict_step(self, batch, batch_idx):
        return self._step(batch, "pred", batch_idx)
    def test_step(self, batch, batch_idx):
        return self._step(batch, "test", batch_idx)

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
    def on_after_backward(self):
        if self.metrics_cfg.log_grad_norm:
            self.log(
                "backbone_grad_norm",
                calculate_norm_grad(self.model.backbone),
                on_step=True,
            )
            self.log(
                "head_grad_norm", calculate_norm_grad(self.model.head), on_step=True
            )

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch"""
        avg_loss = self.trainer.callback_metrics.get("loss_val_epoch", None)
        if avg_loss is not None:
            self.val_losses.append(
                avg_loss.item() if hasattr(avg_loss, "item") else float(avg_loss)
            )
            print(f"Epoch {self.current_epoch} - Val Loss: {self.val_losses[-1]:.6f}")
        mAP_score = self.mAP_score["val"].compute()
        self.log_dict(
            {
                "mAP_score_val": mAP_score,
            }
        )
        print(f"Epoch {self.current_epoch} - mAP: {mAP_score}")
        self.mAP_score["val"].reset()
        if self.use_nms:
            mAP_score_nms = self.mAP_score_nms["val"].compute()
            self.log_dict(
                {
                    "mAP_score_nms": mAP_score_nms,
                }
            )
            print(f"Epoch {self.current_epoch} - mAP_nms: {mAP_score_nms}")
            self.mAP_score_nms["val"].reset()
    def on_test_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get("loss_val_epoch", None)
        if avg_loss is not None:
            self.val_losses.append(
                avg_loss.item() if hasattr(avg_loss, "item") else float(avg_loss)
            )
            print(f"Epoch {self.current_epoch} - Val Loss: {self.val_losses[-1]:.6f}")
        mAP_score = self.mAP_score["test"].compute()
        self.log_dict(
            {
                "mAP_score_test": mAP_score,
            }
        )
        print(f"Epoch {self.current_epoch} - mAP: {mAP_score}")
        self.mAP_score["test"].reset()
        if self.use_nms:
            mAP_score_nms = self.mAP_score_nms["test"].compute()
            self.log_dict(
                {
                    "mAP_score_nms": mAP_score_nms,
                }
            )
            print(f"Epoch {self.current_epoch} - mAP_nms: {mAP_score_nms}")
            self.mAP_score_nms["test"].reset()

    def _step(self, batch, kind, batch_idx):
        images=torch.stack(batch["image"])
        images_paths=batch["image_path"]
        targets=[{"boxes": boxes, "labels": labels} for boxes, labels  in zip(batch["boxes"], batch["labels"])]
        if(kind == "train" or kind=="val" or kind=="test"):
            self.model.train()
            with torch.set_grad_enabled(kind == "train"):
                preds = self.model(images, targets)
                loss = self.criterion(preds, targets)
        if kind=="val" or kind=="test":
            self.model.eval()
            with torch.no_grad():
                preds = self.model(images, targets)
            self.mAP_score[kind].update(preds, targets)
            if self.use_nms:
                preds = nms(preds, iou_thr=self.nms_thr)
                self.mAP_score_nms[kind].update(preds, targets)
            if self.batch_val_show and batch_idx==0:
                self.logger.experiment.add_images(f"{kind}_images",torch.stack(drawbboxes(batch["image"], preds)), global_step=self.global_step)
        if kind=="pred":
            self.model.eval()
            with torch.no_grad():
                preds = self.model(images, targets)
            return images_paths, preds
            #TODO Reshape back the points and log images using batch_idx
        self.log(f"loss_{kind}", loss.item(), on_step=True, on_epoch=True)
        return loss
