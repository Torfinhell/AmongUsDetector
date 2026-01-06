from dataclasses import dataclass, field
from src.data_module import DatasetCreationConfig, DataModuleConfig
from src.models.models_config import ModelFcosPretrainedConfig
from src.metrics import MetricConfig
from typing import Optional


@dataclass
class TrainingConfig:
    num_epochs: int
    val_epoch_len: int
    train_epoch_len: int
    seed: int
    logger_name: str
    logger: Optional[str]  # other variant is wandb
    grad_acum_scheduling: dict[str, int]
    swa_epoch_start: int
    swa_lrs: float


@dataclass
class ModelTrainConfig:
    training_cfg: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            num_epochs=300,
            val_epoch_len=50,
            train_epoch_len=500,
            seed=1,
            logger_name="fcos_training",
            logger="tensorboard",
            grad_acum_scheduling={0: 1},  # TODO
            swa_epoch_start=250,
            swa_lrs=5e-3,
        )
    )

    model_cfg: ModelFcosPretrainedConfig = field(
        default_factory=lambda: ModelFcosPretrainedConfig(
            num_classes=18,
            heads_learning_rate=5e-2,
            backbone_learning_rate=5e-3,
            step_size=3,
            weight_decay=0.001,  # TODO
            gamma=0.98,
            num_anchors=1,
            head_in_channels=256,
            min_size=600,
            max_size=800,
            use_nms=False,
            nms_thr=0.8,
            score_thresh=0.1,
            backbone_layers=5,
            detections_per_img=100,
            topk_candidates=1000,
        )
    )

    creation_cfg: DatasetCreationConfig = field(
        default_factory=lambda: DatasetCreationConfig(
            destination_folder="data/image_train",
            background_folder=None,
            num_figures=20,
            num_generations=5000,
            augment=True,
            draw_bbox=False,
            figure_size_range=(40, 200),
            generate_every_epoch=3,
        )
    )

    datamodule_cfg: DataModuleConfig = field(
        default_factory=lambda: DataModuleConfig(
            generate_new=True,
            train_split=0.8,
            batch_size=12,
            num_workers=4,
        )
    )
    metric_cfg: MetricConfig = field(
        default_factory=lambda: MetricConfig(mAP_iou_thr=0.9, log_grad_norm=True)
    )
