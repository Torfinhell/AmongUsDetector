from dataclasses import dataclass, field
from typing import Annotated, Optional

from cyclopts import Parameter

from src.configs.all_configs import (
    DataModuleConfig,
    MetricConfig,
    ModelFcosPretrainedConfig,
    TransformConfig,
)


@dataclass
class VideoConfig:
    seed: Annotated[int, Parameter(name="--seed")] = 1
    checkpoint: Annotated[
        str,
        Parameter(
            name="--checkpoint",
        ),
    ] = None
    accelerator: Annotated[str, Parameter(name="--accelerator")] = "gpu"
    show_real_time: Annotated[bool, Parameter(name="--show")] = False  # TODO
    calculate_metrics: Annotated[bool, Parameter(name="--calc_metrics")] = False  # TODO
    frames_per_sec: Annotated[Optional[bool], Parameter(name="--calc_metrics")] = False


@dataclass
class ModelVideoConfig:
    video_cfg: VideoConfig = field(default_factory=lambda: VideoConfig())
    model_cfg: ModelFcosPretrainedConfig = field(
        default_factory=lambda: ModelFcosPretrainedConfig()
    )
    creation_cfg = None
    datamodule_cfg: DataModuleConfig = field(default_factory=lambda: DataModuleConfig())
    metric_cfg: MetricConfig = field(default_factory=lambda: MetricConfig())
    transform_cfg: TransformConfig = field(default_factory=lambda: TransformConfig())
