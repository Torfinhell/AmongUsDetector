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
        self.mAP_score = {"generated":PRauc(self.metrics_cfg), "test":PRauc(self.metrics_cfg)}
        self.use_nms = cfg.model_cfg.use_nms
        self.nms_thr = cfg.model_cfg.nms_thr
        self.batch_val_show=cfg.datamodule_cfg.batch_val_show
        if self.use_nms:
            self.mAP_score_nms = {"generated":PRauc(self.metrics_cfg), "test":PRauc(self.metrics_cfg)}
        self.dataloader_names = {0: "generated", 1: "test"}
        self.losses_names=['classification', 'bbox_regression', 'bbox_ctrness', "total_loss"]
        

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

    def validation_step(self, batch, batch_idx, dataloader_idx:int=0):
        return self._step(batch, f"val_{self.dataloader_names[dataloader_idx]}", batch_idx)
    def predict_step(self, batch, batch_idx):
        return self._step(batch, "pred", batch_idx)

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        for loss_kind in self.losses_names:
                avg_loss = self.trainer.callback_metrics.get(f"train/{loss_kind}", None)
                if avg_loss is not None:
                    print(f"Epoch {self.current_epoch} - {loss_kind} Train Loss: {avg_loss.item():.6f}")
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
        for name_val_ds in self.dataloader_names.values():
            for loss_kind in self.losses_names:
                avg_loss = self.trainer.callback_metrics.get(f"val_{name_val_ds}/{loss_kind}", None)
                if avg_loss is not None:
                    print(f"Epoch {self.current_epoch} - {loss_kind} Val_{name_val_ds} Loss: {avg_loss.item():.6f}")
            mAP_score = self.mAP_score[name_val_ds].compute()
            self.log_dict(
                {
                    f"mAP_score_val_{name_val_ds}": mAP_score,
                },
                add_dataloader_idx=False
            )
            print(f"Epoch {self.current_epoch} - mAP_{name_val_ds}: {mAP_score}")
            self.mAP_score[name_val_ds].reset()
            if self.use_nms:
                mAP_score_nms = self.mAP_score_nms[name_val_ds].compute()
                self.log_dict(
                    {
                        f"mAP_score_nms_{name_val_ds}": mAP_score_nms,
                    },
                    add_dataloader_idx=False
                )
                print(f"Epoch {self.current_epoch} - mAP_nms: {mAP_score_nms}")
                self.mAP_score_nms[name_val_ds].reset()
    def _step(self, batch, kind, batch_idx):
        images=torch.stack(batch["image"])
        images_paths=batch["image_path"]
        targets=[{"boxes": boxes, "labels": labels} for boxes, labels  in zip(batch["boxes"], batch["labels"])]
        if(kind == "train" or kind.startswith("val")):
            self.model.train()
            with torch.set_grad_enabled(kind == "train"):
                losses = self.model(images, targets)
                losses["total_loss"] = self.criterion(losses, targets)
                for loss_kind in self.losses_names:
                    self.log(f"{kind}/{loss_kind}", losses[loss_kind], on_step=True, on_epoch=True, add_dataloader_idx=False)
        if kind.startswith("val"):
            self.model.eval()
            with torch.no_grad():
                preds = self.model(images, targets)
            self.mAP_score[kind.split("_")[-1]].update(preds, targets)
            if self.use_nms:
                preds = nms(preds, iou_thr=self.nms_thr)
                self.mAP_score_nms[kind.split("_")[-1]].update(preds, targets)
            if self.batch_val_show and batch_idx==0:
                self.logger.experiment.add_images(f"{kind}_images",torch.stack(drawbboxes(batch["image"], preds)), global_step=self.global_step)
        if kind=="pred":
            self.model.eval()
            with torch.no_grad():
                preds = self.model(images, targets)
            return images_paths, preds
            #TODO Reshape back the points and log images using batch_idx
        return losses["total_loss"]
