from dataclasses import dataclass, field
from src.configs.all_configs import ModelFcosPretrainedConfig,DatasetCreationConfig, DataModuleConfig, MetricConfig, TransformConfig
from typing import Optional

@dataclass
class TrainingConfig:
    num_epochs:int=300
    val_epoch_len:Optional[int]=50
    train_epoch_len:Optional[int]=500
    seed:int=1
    logger_name:str="fcos_test"
    logger:str="tensorboard"
    grad_acum_scheduling:dict[str, int]=field(default_factory=lambda: {"0": 1}) # TODO
    swa_epoch_start:Optional[int]=250
    swa_lrs:Optional[float]=5e-3
@dataclass
class ModelTrainConfig:
    training_cfg: TrainingConfig=field(default_factory=lambda: TrainingConfig())

    model_cfg: ModelFcosPretrainedConfig=field(default_factory=lambda: ModelFcosPretrainedConfig())

    creation_cfg: DatasetCreationConfig=field(default_factory=lambda: DatasetCreationConfig())

    datamodule_cfg: DataModuleConfig=field(default_factory=lambda: DataModuleConfig())
    metric_cfg: MetricConfig=field(default_factory=lambda: MetricConfig())
    transform_cfg:TransformConfig=field(default_factory=lambda:TransformConfig())
