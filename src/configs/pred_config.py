from dataclasses import dataclass, field
from src.configs.all_configs import ModelFcosPretrainedConfig, MetricConfig, DatasetCreationConfig, DataModuleConfig


@dataclass
class InferenceConfig:
    seed: int
    input_floder: str = "data/image_train"
    output_floder: str = "data/output_image_train"
    confidence: float = 0.5  # Confidence threshold for drawing boxes from 0 to 1
    checkpoint: str = "checkpoints/fcos-epoch=61-loss_val=0.7329.ckpt"


@dataclass
class ModelPredConfig:
    inference_cfg: InferenceConfig = field(
        default_factory=lambda: InferenceConfig(
            seed=1,
            input_floder = "data/image_train",
            output_floder="data/output_image_train",
            confidence=0.5,
            checkpoint="checkpoints/fcos-epoch=61-loss_val=0.7329.ckpt",
        )
    )

    model_cfg: ModelFcosPretrainedConfig = field(
        default_factory=lambda: ModelFcosPretrainedConfig(
            num_classes=18,
            num_anchors=1,  # TODO
            head_in_channels=256,
            min_size=600,  # TODO
            max_size=800,
            use_nms=False,
            nms_thr=0.8,
            score_thresh=0.3,
            backbone_layers=5,
            detections_per_img=100,
            topk_candidates=1000,
        )
    )

    creation_cfg = None

    datamodule_cfg: DataModuleConfig = field(
        default_factory=lambda: DataModuleConfig(
            generate_new=False,
            batch_size=12,
            num_workers=4,
            val_num_generations=None,
            train_num_generations=None,
            image_val_folder=None,
            image_train_folder=None,
            image_test_data=None,
            image_pred_data="data/image_train/val",
            generate_every_epoch=3,
        )
    )
    metric_cfg: MetricConfig = field(
        default_factory=lambda: MetricConfig(
            mAP_iou_thr=0.9,
        )
    )
