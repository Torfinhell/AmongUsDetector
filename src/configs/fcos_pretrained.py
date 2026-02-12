from dataclasses import dataclass, field
from typing import Annotated, Optional

from cyclopts import Parameter

from src.configs.all_configs import (
    DataModuleConfig,
    DatasetCreationConfig,
    MetricConfig,
    ModelFcosPretrainedConfig,
    TransformConfig,
)


@dataclass
class TrainingConfig:
    num_epochs: Annotated[int, Parameter(name="--num_epochs")] = 300
    val_epoch_len: Annotated[Optional[int], Parameter(name="--val_epoch_len")] = 50
    train_epoch_len: Annotated[Optional[int], Parameter(name="--train_epoch_len")] = 500
    seed: Annotated[int, Parameter(name="--seed")] = 1
    logger_name: Annotated[str, Parameter(name="--logger_name")] = "fcos_test"
    logger: str = "tensorboard"
    grad_acum_scheduling: dict[str, int] = field(default_factory=lambda: {"0": 1})
    swa_epoch_start: Annotated[Optional[int], Parameter(name="--swa_epoch_start")] = 250
    swa_lrs: Annotated[Optional[float], Parameter(name="--swa_lrs")] = 5e-3
    finetune_chk: Annotated[Optional[str], Parameter(name="--finetune_chk")] = None


@dataclass
class ModelTrainConfig:
    training_cfg: TrainingConfig = field(default_factory=lambda: TrainingConfig())

    model_cfg: ModelFcosPretrainedConfig = field(
        default_factory=lambda: ModelFcosPretrainedConfig()
    )

    creation_cfg: DatasetCreationConfig = field(
        default_factory=lambda: DatasetCreationConfig()
    )

    datamodule_cfg: DataModuleConfig = field(default_factory=lambda: DataModuleConfig())
    metric_cfg: MetricConfig = field(default_factory=lambda: MetricConfig())
    transform_cfg: TransformConfig = field(default_factory=lambda: TransformConfig())
