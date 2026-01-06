import lightning as L
from src.data_module import AmongUsDatamodule
from src.models import ModelFcosPretrained
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.configs import ModelTrainConfig
from cyclopts import App
from src.utils import set_seed

app = App(name="Define Config for training:")


@app.command
def train_fcos(cfg: ModelTrainConfig = ModelTrainConfig()):
    training_cfg = cfg.training_cfg
    set_seed(training_cfg.seed)

    # initialize Datamodule
    data_module = AmongUsDatamodule(cfg.datamodule_cfg, cfg.creation_cfg)
    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir="logs", name=training_cfg.logger_name, version=None
    )
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="loss_val",
        mode="min",
        save_top_k=3,
        dirpath="checkpoints",
        filename="fcos-{epoch:02d}-{loss_val:.4f}",
    )
    # Gradient Norm Output
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=training_cfg.num_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
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
