import lightning as L
from torch.utils.data import DataLoader
from src.datasets import AmongUsImagesDataset
from scripts.generate import generate
from dataclasses import dataclass
from src.models import ModelFcos
from src.transforms import FcosTransform
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

# Set float32 matmul precision for better GPU utilization
torch.set_float32_matmul_precision("medium")


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    FCOS requires images as a list and targets as a list of dicts.
    Handles variable-sized bounding boxes across batch.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


@dataclass
class ModelConfig:
    batch_size: int
    num_epochs: int
    num_workers: int = 4
    augment: bool = True


def train_fcos(cfg: ModelConfig):
    # background_folder="background_images" #for later
    generate(
        "../data/image_train",
        background_folder=None,
        num_generations=100,
        num_figures=5,
        augment=True,
        random_color=True,
        draw_bbox=True,
        figure_size_range=(80, 150),
    )
    # Create training dataset
    train_dataset = AmongUsImagesDataset(
        path_to_data="data/image_train",
        augment=cfg.augment,
        transform=FcosTransform() if cfg.augment else None,
        partition="train",
        train_split=0.8,
    )

    # Create validation dataset (no augmentation for validation)
    val_dataset = AmongUsImagesDataset(
        path_to_data="data/image_train",
        augment=False,
        transform=(
            FcosTransform(p=0.0) if cfg.augment else None
        ),  # No augmentation, just resize/normalize
        partition="val",
        train_split=0.8,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir="logs", name="fcos_training", version=None)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="loss_val",
        mode="min",
        save_top_k=3,
        dirpath="checkpoints",
        filename="fcos-{epoch:02d}-{loss_val:.4f}",
    )

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=cfg.num_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Clip gradients to prevent NaN
        enable_progress_bar=True,
    )
    model = ModelFcos()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return trainer, model


if __name__ == "__main__":
    cfg = ModelConfig(batch_size=8, num_epochs=10, num_workers=4, augment=True)
    trainer, model = train_fcos(cfg)

    # Print final epoch losses
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best checkpoint saved in: checkpoints/")
    print(f"Logs saved in: logs/")
    print(f"\nTo view training metrics:")
    print(f"  tensorboard --logdir=logs")
    print("=" * 80 + "\n")
