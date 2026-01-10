import lightning as L
from src.data_module import AmongUsDatamodule
from src.models import ModelFcosPretrained
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch  import seed_everything
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from src.configs import ModelTrainConfig
from cyclopts import App
from src.utils import  FineTuneLearningRateFinder

app = App(name="Define Config for training:")


@app.command
def train_fcos(cfg: ModelTrainConfig = ModelTrainConfig()):
    training_cfg = cfg.training_cfg
    seed_everything(training_cfg.seed)

    # initialize Datamodule
    data_module = AmongUsDatamodule(cfg.datamodule_cfg, cfg.creation_cfg, cfg.transform_cfg)
    # Setup TensorBoard logger

    tb_logger = TensorBoardLogger(
        save_dir="logs", name=training_cfg.logger_name, version=None
    )
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="mAP_score",
        mode="max",
        save_top_k=3,
        dirpath="checkpoints",
        filename="fcos-{epoch:02d}-{loss_val:.4f}",
        save_last=True,
        enable_version_counter=True,
    )
    grad_acum = GradientAccumulationScheduler(
        scheduling={int(k): v for k, v in training_cfg.grad_acum_scheduling.items()}
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    swa = StochasticWeightAveraging(
        swa_lrs=training_cfg.swa_lrs,
        swa_epoch_start=250,
        annealing_epochs=10,
        annealing_strategy="cos",
    )
    # Gradient Norm Output
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=training_cfg.num_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor, swa, grad_acum],
        log_every_n_steps=10,
        gradient_clip_val=6.0,
        enable_progress_bar=True,
        limit_train_batches=training_cfg.train_epoch_len,
        limit_val_batches=training_cfg.val_epoch_len,
    )
    model = ModelFcosPretrained(cfg)
    trainer.fit(model=model, datamodule=data_module)
    return trainer, model


if __name__ == "__main__":
    app()
