from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from collections import defaultdict
import PIL.Image
import numpy as np
import torch
from .utils import color_to_ind
import torchvision.transforms.v2 as v2


class AmongUsImagesDataset(Dataset):

    def __init__(
        self,
        path_to_data,
        transform,  # transform should not be None
        partition="train",
        train_split=0.8,
    ):
        """
        Dataset for Among Us character detection with bounding boxes.

        Args:
            path_to_data: Path to data directory containing 'images' folder and 'images.csv'
            augment: Whether to apply augmentation (currently unused, use transform instead)
            transform: Transform to apply (e.g., FcosTransform instance)
            partition: 'train' or 'val' - which partition to use
            train_split: float between 0 and 1, percentage of data for training (default 0.8 = 80% train, 20% val)
        """
        self.path_to_images = Path(path_to_data) / "images"
        self.partition = partition
        self.train_split = train_split
        self.transform = transform
        self.csv_path = Path(path_to_data) / "images.csv"
        self.update_data()

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
            image, bboxes = self.transform(image=image, bboxes=bboxes)
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
        target = {"boxes": boxes, "labels": labels}
        return image, target

    def __len__(self):
        return len(self.images_paths)

    def update_data(self):
        all_images = sorted(
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
        if self.csv_path.exists():
            images_info = pd.read_csv(self.csv_path)
            for _, row in images_info.iterrows():
                self.bboxes[row.iloc[0]].append(list(row.iloc[1:5]))
                self.labels[row.iloc[0]].append(color_to_ind[row.iloc[5]])

        # Split into train/val
        num_train = int(len(all_images) * self.train_split)
        if self.partition == "train":
            self.images_paths = all_images[:num_train]
        elif self.partition == "val":
            self.images_paths = all_images[num_train:]
        else:
            raise ValueError(
                f"partition must be 'train' or 'val', got {self.partition}"
            )
