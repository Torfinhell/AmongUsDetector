from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DataModuleConfig:
    generate_new: bool = True
    train_split: float = 0.8
    batch_size: int = 3
    num_workers: int = 4


@dataclass
class DatasetCreationConfig:
    destination_folder: str = ""
    background_folder: Optional[str] = None
    num_generations: int = 10
    num_figures: int = 3
    augment: bool = False
    random_color: bool = True
    draw_bbox: bool = False
    figure_size_range: Tuple[int, int] = (80, 200)
    generate_every_epoch: bool = False
