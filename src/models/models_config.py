from dataclasses import dataclass


@dataclass
class ModelFcosPretrainedConfig:
    num_classes: int = 18
    heads_learning_rate: float = 4e-2
    backbone_learning_rate: float = 1e-3
    step_size: int = 3
    weight_decay: float = 0.0005
    gamma: float = 0.98
    num_anchors: int = 1
    head_in_channels: int = 256
    min_size: int = 600
    max_size: int = 800
    use_nms: bool = True
    nms_thr: float = 0.8
    score_thresh: float = 0.1
    backbone_layers: int = 5
    detections_per_img: int = 100
    topk_candidates: int = 1000
