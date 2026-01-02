from dataclasses import dataclass, field
from src.data_module import DatasetCreationConfig, DataModuleConfig
from src.models import ModelFcosPretrainedConfig


# from typing import Union
# from train import TrainingConfig
@dataclass
class TrainingConfig:
    num_epochs: int
    val_epoch_len: int
    train_epoch_len: int
    seed: int


@dataclass
class ModelConfig:
    training_cfg: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            num_epochs=300,
            val_epoch_len=50,
            train_epoch_len=500,
            seed=1,
        )
    )

    model_cfg: ModelFcosPretrainedConfig = field(
        default_factory=lambda: ModelFcosPretrainedConfig(
            num_classes=18,
            learning_rate=4e-3,
            step_size=3,
            weight_decay=0.0005,
            gamma=0.98,
        )
    )

    creation_cfg: DatasetCreationConfig = field(
        default_factory=lambda: DatasetCreationConfig(
            destination_folder="data/image_train",
            background_folder=None,
            num_figures=1,
            num_generations=100,
            augment=True,
            draw_bbox=False,
            figure_size_range=(80, 200),
        )
    )

    datamodule_cfg: DataModuleConfig = field(
        default_factory=lambda: DataModuleConfig(
            generate_new=True,
            train_split=0.8,
            batch_size=3,
            num_workers=4,
        )
    )
