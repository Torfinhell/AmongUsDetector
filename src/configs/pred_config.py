from dataclasses import dataclass, field
from src.configs.all_configs import ModelFcosPretrainedConfig, MetricConfig, TransformConfig, DataModuleConfig
from cyclopts import Parameter
from typing import Optional, Annotated

@dataclass
class InferenceConfig:
    seed: Annotated[int, Parameter(name="--seed")]=1
    checkpoint: Annotated[Optional[str], Parameter(name="--checkpoint")]=None


@dataclass
class ModelPredConfig:
    inference_cfg: InferenceConfig = field(
        default_factory=lambda: InferenceConfig()
    )

    model_cfg: ModelFcosPretrainedConfig = field(
        default_factory=lambda: ModelFcosPretrainedConfig()
    )

    creation_cfg = None

    datamodule_cfg: DataModuleConfig = field(
        default_factory=lambda: DataModuleConfig()
    )
    metric_cfg: MetricConfig = field(
        default_factory=lambda: MetricConfig(
        )
    )
    transform_cfg:TransformConfig=field(default_factory=lambda:TransformConfig())
