from dataclasses import dataclass, field
from src.configs.all_configs import ModelFcosPretrainedConfig, MetricConfig, DatasetCreationConfig, DataModuleConfig

@dataclass
class InferenceConfig:
    seed: int=1
    input_floder: str = "data/image_train"
    output_floder: str = "data/output_image_train"
    confidence: float = 0.5  # Confidence threshold for drawing boxes from 0 to 1
    checkpoint: str = "checkpoints/fcos-epoch=61-loss_val=0.7329.ckpt"


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
        default_factory=lambda: DataModuleConfig() # image_pred_data="data/image_train/val"
    )
    metric_cfg: MetricConfig = field(
        default_factory=lambda: MetricConfig(
        )
    )
