from dataclasses import dataclass, field
from src.data_module import DatasetCreationConfig, DataModuleConfig
from src.models.models_config import ModelFcosPretrainedConfig
from src.metrics import MetricConfig
from typing import Optional


@dataclass
class InferenceConfig:
    seed: int
    input_floder: str = ("data/image_train",)
    output_floder: str = "data/output_image_train"
    images_info: Optional[str] = None
    confidence: float = 0.5  # Confidence threshold for drawing boxes from 0 to 1
    checkpoint: str = "checkpoints/fcos-epoch=11-loss_val=0.7546.ckpt"


@dataclass
class ModelPredConfig:
    inference_cfg: InferenceConfig = field(
        default_factory=lambda: InferenceConfig(
            seed=1,
            confidence=0.5,
            output_floder="data/output_image_train",
            images_info=None,
            confidence=0.5,
            checkpoint="checkpoints/fcos-epoch=11-loss_val=0.7546.ckpt",
        )
    )

    model_cfg: ModelFcosPretrainedConfig = field(
        default_factory=lambda: ModelFcosPretrainedConfig(
            num_classes=18,
            learning_rate=8e-3,
            step_size=3,
            weight_decay=0.0005,
            gamma=0.98,
            num_anchors=1,
            head_in_channels=256,
            min_size=600,
            max_size=800,
            use_nms=False,
            nms_thr=0.8,
            score_thresh=0.3,
            backbone_layers=5,
            detections_per_img=100,
            topk_candidates=1000,
        )
    )

    creation_cfg: DatasetCreationConfig = field(
        default_factory=lambda: DatasetCreationConfig(
            destination_folder="data/image_train",
            background_folder=None,
            num_figures=15,
            num_generations=100000,
            augment=True,
            draw_bbox=False,
            figure_size_range=(40, 200),
            generate_every_epoch=False,
        )
    )

    datamodule_cfg: DataModuleConfig = field(
        default_factory=lambda: DataModuleConfig(
            generate_new=False,
            train_split=0.8,
            batch_size=12,
            num_workers=4,
        )
    )
    metric_cfg: MetricConfig = field(
        default_factory=lambda: MetricConfig(
            iou_thr=0.9,
        )
    )
