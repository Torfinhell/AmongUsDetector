import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import torch
import torchvision.transforms.v2 as v2
from torch import nn
from torch.utils.data import Dataset

from src.utils import color_to_ind


class AmongUsImagesDataset(Dataset):

    def __init__(
        self,
        path_to_images,
        path_to_csv=None,
        transform=None,  # transform should not be None
    ):
        """
        Dataset for Among Us character detection with bounding boxes.

        Args:
            path_to_data: Path to data directory containing 'images' folder and 'images.csv'
            augment: Whether to apply augmentation (currently unused, use transform instead)
            transform: Transform to apply (e.g., FcosTransform instance)
        """
        self.path_to_images = Path(path_to_images)
        assert os.path.exists(
            self.path_to_images
        ), f"Path to {self.path_to_images} should exist to load images"
        self.transform = transform
        self.images_paths = sorted(
            [
                path
                for path in self.path_to_images.iterdir()
                if path.suffix.lower() in [".jpg", ".png", ".jpeg"]
            ]
        )

        self.bboxes = defaultdict(
            list
        )  # path to array of bboxes [x_min, y_min, x_max, y_max]
        self.labels = defaultdict(list)
        # Load bounding boxes from CSV
        if path_to_csv is not None:
            csv_path = Path(path_to_csv)
            if csv_path.exists():
                images_info = pd.read_csv(csv_path)
                for _, row in images_info.iterrows():
                    self.bboxes[row.iloc[0]].append(list(row.iloc[1:5]))
                    self.labels[row.iloc[0]].append(color_to_ind[row.iloc[5]])

    def __getitem__(self, idx):
        """
        Returns:
            image: torch tensor of shape (3, H, W)
            target: dict with keys 'boxes' and 'labels' for FCOS
        """
        image_path = self.images_paths[idx]
        bboxes = self.bboxes[image_path.name]
        labels = self.labels[image_path.name]
        # Load image as RGB numpy array
        image = np.array(PIL.Image.open(image_path).convert("RGB"))
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes)
            image, bboxes = transformed
        else:
            image = v2.ToImage()(image)
            image = v2.ToDtype(torch.float32, scale=True)(image)
        # Convert to FCOS format: dict with 'boxes' and 'labels'
        if len(bboxes) > 0:
            boxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.images_paths)

    def show_dataset(num_images: int, model: nn.Module):
        pass
