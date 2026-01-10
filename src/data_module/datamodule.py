from src.data_module.dataset import AmongUsImagesDataset
from src.data_module.utils import collate_fn
import lightning as L
from torch.utils.data import DataLoader
from src.data_module.generate import generate_data
from src.configs import DataModuleConfig, DatasetCreationConfig, TransformConfig
from copy import deepcopy
from src.transforms import FcosTransform


class AmongUsDatamodule(L.LightningDataModule):
    def __init__(
        self,
        datamodule_cfg: DataModuleConfig,
        creation_cfg: DatasetCreationConfig,
        transform_cfg:TransformConfig
    ):
        super().__init__()
        self.creation_cfg = deepcopy(creation_cfg)
        self.image_train_data = datamodule_cfg.image_train_folder
        self.image_val_data = datamodule_cfg.image_val_folder
        self.image_test_data = datamodule_cfg.image_test_folder
        self.batch_size = datamodule_cfg.batch_size
        self.num_workers = datamodule_cfg.num_workers
        self.train_generations = datamodule_cfg.train_num_generations
        self.val_generations = datamodule_cfg.val_num_generations
        self.test_data = datamodule_cfg.image_test_folder
        self.predict_data=datamodule_cfg.image_pred_data
        self.generate_new = datamodule_cfg.generate_new
        self.generate_every_epoch = datamodule_cfg.generate_every_epoch
        self.transform_cfg=transform_cfg

    def setup(self, stage):
        if stage == "fit" and self.image_train_data is not None:
            if self.generate_new and self.generate_every_epoch is None:
                self.creation_cfg.destination_folder = self.image_train_data
                self.creation_cfg.num_generations = self.train_generations
                generate_data(self.creation_cfg)
            self.train_dataset = AmongUsImagesDataset(
                path_to_images=self.image_train_data+"/images",
                path_to_csv=self.image_train_data+"/images.csv",
                transform=FcosTransform(self.transform_cfg, part="train")
                )
        if stage == "fit" and self.image_val_data is not None:
            if self.generate_new:
                self.creation_cfg.destination_folder = self.image_val_data
                self.creation_cfg.num_generations = self.val_generations
                generate_data(self.creation_cfg)
            self.val_dataset = AmongUsImagesDataset(
                path_to_images=self.image_val_data+"/images",
                path_to_csv=self.image_val_data+"/images.csv",
                transform=FcosTransform(self.transform_cfg, part="val")
            )
        if stage == "test" and self.image_test_data is not None:
            self.test_dataset = AmongUsImagesDataset(
                path_to_images=self.image_test_data+"/images",
                path_to_csv=self.image_test_data+"/images.csv",
                transform=FcosTransform(self.transform_cfg, part="test")
            )
        if stage == "predict":
            self.pred_dataset = AmongUsImagesDataset(
                path_to_data=self.predict_data,
                transform=FcosTransform(self.transform_cfg, part="pred")
            )

    def train_dataloader(self):
        if (
            self.generate_new
            and self.generate_every_epoch is not None
            and self.trainer.current_epoch % self.generate_every_epoch == 0
        ):
            self.creation_cfg.destination_folder = self.image_train_data
            self.creation_cfg.num_generations = self.train_generations
            generate_data(self.creation_cfg)
            self.train_dataset = AmongUsImagesDataset(
                path_to_images=self.image_train_data+"/images",
                path_to_csv=self.image_train_data+"/images.csv",
                transform=FcosTransform(self.transform_cfg, part="train")
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )

    # TODO Pred Dataloader?
