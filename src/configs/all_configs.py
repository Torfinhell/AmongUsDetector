from dataclasses import dataclass
from typing import Optional
from cyclopts import Parameter
from typing import Optional, Tuple, Annotated
@dataclass
class ModelFcosPretrainedConfig:
    num_classes:int=18
    heads_learning_rate:float=8e-3
    backbone_learning_rate:float=8e-3
    step_size:int=3
    weight_decay:float=0.0005  # TODO
    gamma:float=0.98
    num_anchors:int=1
    head_in_channels:int=256
    min_size:int=600 #TODO
    max_size:int=800
    use_nms:bool=False
    nms_thr:float=0.8
    score_thresh:float=0.1
    backbone_layers:int=5
    detections_per_img:int=100
    topk_candidates:int=1000

@dataclass
class MetricConfig:
    mAP_iou_thr: float = 0.9  
    log_grad_norm: bool = True

@dataclass
class DataModuleConfig:
    generate_new:bool=True
    batch_size:Annotated[int, Parameter(name="--batch_size")]=12
    num_workers:int=4
    val_num_generations:Annotated[int, Parameter(name="--val_num_gen")]=1000
    train_num_generations:Annotated[int, Parameter(name="--train_num_gen")]=4000
    image_val_folder:Annotated[Optional[str], Parameter(name="--val_folder")]=None
    image_train_folder:Annotated[Optional[str], Parameter(name="--train_folder")]=None
    image_test_folder:Annotated[Optional[str], Parameter(name="--test_folder")]=None
    image_pred_data:Annotated[Optional[str],Parameter(name="--pred_data")]=None
    generate_every_epoch:Annotated[Optional[int],Parameter(name="--pred_data")]=3


@dataclass
class DatasetCreationConfig:
    destination_folder: Annotated[Optional[str], Parameter(name="--dest_folder")]=None #data/image_train/val or data/image_train/train as an example
    background_folder: Annotated[Optional[str], Parameter(name="--background_folder")] = None
    num_generations: Annotated[Optional[int], Parameter(name="--num_generation")] = None
    num_figures: Annotated[int, Parameter(name="--num_figures")] = 20
    augment_figure: Annotated[bool,Parameter(name="--augment_figure")]  = True
    augment_mask: Annotated[bool,Parameter(name="--augmet_mask")]  = True
    draw_bbox: Annotated[bool, Parameter(name="--draw_bbox")] = False
    figure_size_range: Tuple[int, int] = (40, 200)

