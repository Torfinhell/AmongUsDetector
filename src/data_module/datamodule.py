from src.data_module.dataset import AmongUsImagesDataset
from src.data_module.utils import collate_fn
import lightning as L
from torch.utils.data import DataLoader
from src.data_module.generate import generate
from dataclasses import dataclass
from src.data_module.generate import DatasetCreationConfig


@dataclass
class DataModuleConfig:
    generate_new: bool = True
    train_split: float = 0.8
    batch_size: int = 3
    num_workers: int = 4


class AmongUsDatamodule(L.LightningDataModule):
    def __init__(
        self,
        datamodule_cfg: DataModuleConfig = DataModuleConfig(),
        creation_cfg: DatasetCreationConfig = DatasetCreationConfig(),
    ):
        super().__init__()
        self.creation_cfg = creation_cfg
        self.datamodule_cfg = datamodule_cfg

    def setup(self, stage):
        # TODO stage???
        if self.datamodule_cfg.generate_new:
            generate(self.creation_cfg)
        self.train_dataset = AmongUsImagesDataset(
            path_to_data=self.creation_cfg.destination_folder,
            transform=None,  # TODO FCOS Transform
            partition="train",
            train_split=self.datamodule_cfg.train_split,
        )
        self.val_dataset = AmongUsImagesDataset(
            path_to_data=self.creation_cfg.destination_folder,
            transform=None,  # TODO FCOS Transform # No augmentation, just resize/normalize #TODO calculate mean and std?
            partition="val",
            train_split=self.datamodule_cfg.train_split,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.datamodule_cfg.batch_size,
            num_workers=self.datamodule_cfg.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.datamodule_cfg.batch_size,
            num_workers=self.datamodule_cfg.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )

    # TODO Pred Dataloader?
