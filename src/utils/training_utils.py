import lightning as pl
from lightning.pytorch.callbacks import LearningRateFinder


class TestEveryNEpochs(pl.Callback):
    def __init__(self, n_epochs=5):
        self.n_epochs = n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.n_epochs == 0:
            print(f"Running test at epoch {trainer.current_epoch+1}")
            trainer.test(pl_module, datamodule=trainer.datamodule)


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)
