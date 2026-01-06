from dataclasses import dataclass


@dataclass
class MetricConfig:
    mAP_iou_thr: float = 0.3  # for mAP
    log_grad_norm: bool = True
