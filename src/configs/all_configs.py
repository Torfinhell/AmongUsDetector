from dataclasses import dataclass
from typing import Optional
from cyclopts import Parameter
from typing import Optional, Tuple, Annotated
@dataclass
class ModelFcosPretrainedConfig:
    num_classes:int=18
    heads_learning_rate:Annotated[Optional[float], Parameter(name="--head_lr")]=8e-3
    backbone_learning_rate:Annotated[Optional[float], Parameter(name="--backbone_lr")]=8e-3
    step_size:int=3
    weight_decay:float=0.0005  
    gamma:float=0.98
    num_anchors:int=1
    head_in_channels:int=256
    min_size:int=400 
    max_size:int=800
    use_nms:bool=False
    nms_thr:float=0.8
    score_thresh:float=0.1
    backbone_layers:int=5
    detections_per_img:int=20
    topk_candidates:int=1000
    is_pretrained:Annotated[Optional[bool], Parameter(name="--is_pretrained")]=True

@dataclass
class MetricConfig:
    mAP_iou_thr: float = 0.9  
    log_grad_norm: Optional[bool] = True
    log_image_folder:str="data/image_log"
    log_image_num_step:Annotated[Optional[int],Parameter(name="--log_image_num_step")]=100

@dataclass
class DataModuleConfig:
    generate_new:Annotated[Optional[bool],Parameter(name="--generate_new")]=True #TODO
    generate_every_epoch:Annotated[Optional[int],Parameter(name="--generate_every_epoch")]=1 #TODO
    batch_size:Annotated[int, Parameter(name="--batch_size")]=30
    num_workers:Annotated[int, Parameter(name="--num_workers")]=4
    val_num_generations:Annotated[int, Parameter(name="--val_num_gen")]=1500#50*30
    train_num_generations:Annotated[int, Parameter(name="--train_num_gen")]=3000#100*30
    image_val_folder:Annotated[Optional[str], Parameter(name="--val_folder")]=None
    image_train_folder:Annotated[Optional[str], Parameter(name="--train_folder")]=None
    image_test_folder:Annotated[Optional[str], Parameter(name="--test_folder")]=None
    image_pred_data:Annotated[Optional[str],Parameter(name="--pred_data")]=None
    pred_output:Annotated[Optional[str],Parameter(name="--pred_output")]=None
    batch_val_show:Annotated[bool, Parameter(name="--num_val_show")]=True

@dataclass
class DatasetCreationConfig:
    destination_folder: Annotated[Optional[str], Parameter(name="--dest_folder")]=None #data/image_train/val or data/image_train/train as an example
    background_folder: Annotated[Optional[str], Parameter(name="--background_folder")] = None
    num_generations: Annotated[Optional[int], Parameter(name="--num_generation")] = None
    num_figures: Annotated[int, Parameter(name="--num_figures")] = 20
    augment_figure: Annotated[bool,Parameter(name="--augment_figure")]  = True
    augment_mask: Annotated[bool,Parameter(name="--augment_mask")]  = True
    draw_bbox: Annotated[bool, Parameter(name="--draw_bbox")] = False
    bg_shape:Annotated[Tuple[int, int], Parameter(name="--bg_shape")]=(640, 360) #width, height
    width_mean_std:Tuple[float, float]=(41.0,17.75)#for figure
    ratio_mean_std:Tuple[float, float]=(1.342, 0.179)
    width_range:Tuple[float, float]=(20.0, 150.0)

@dataclass
class TransformConfig:
    normalize:bool=False
    height:int=300 
    width:int=300

