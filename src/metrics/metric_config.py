from dataclasses import dataclass


@dataclass
class MetricConfig:
    iou_thr: float = 0.3  # for mAP
