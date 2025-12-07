import albumentations as A
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from collections import defaultdict
import PIL.Image
import numpy as np
import torch


class AmongUsImagesDataset(Dataset):
    def __init__(
        self,
        path_to_data,
        augment=True,
        transform=None,
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
        path_to_images = Path(path_to_data) / "images"
        all_images = sorted(
            [
                path
                for path in path_to_images.iterdir()
                if path.suffix.lower() in [".jpg", ".png", ".jpeg"]
            ]
        )

        self.bboxes = defaultdict(
            list
        )  # path to array of bboxes [x_min, y_min, x_max, y_max]

        # Load bounding boxes from CSV
        csv_path = Path(path_to_data) / "images.csv"
        if csv_path.exists():
            images_info = pd.read_csv(csv_path)
            for _, row in images_info.iterrows():
                self.bboxes[row.iloc[0]].append(list(row.iloc[1:5]))

        # Split into train/val
        num_train = int(len(all_images) * train_split)
        if partition == "train":
            self.images_paths = all_images[:num_train]
        elif partition == "val":
            self.images_paths = all_images[num_train:]
        else:
            raise ValueError(f"partition must be 'train' or 'val', got {partition}")

        self.transform = transform

    def __getitem__(self, idx):
        """
        Returns:
            image: torch tensor of shape (3, H, W)
            target: dict with keys 'boxes' and 'labels' for FCOS
        """
        image_path = self.images_paths[idx]
        bboxes = self.bboxes[image_path.name]

        # Load image as RGB numpy array
        image = np.array(PIL.Image.open(image_path).convert("RGB"))

        # Apply transform
        if self.transform is not None:
            image, bboxes, class_labels = self.transform(image=image, bboxes=bboxes)
        else:
            # Minimal transform: just convert to tensor and normalize
            from albumentations.pytorch import ToTensorV2

            minimal_transform = A.Compose(
                [
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ]
            )
            image = minimal_transform(image=image)["image"]
            class_labels = [0] * len(bboxes)

        # Convert to FCOS format: dict with 'boxes' and 'labels'
        if len(bboxes) > 0:
            boxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(class_labels, dtype=torch.int64)
        else:
            # Handle empty image
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        return image, target

    def __len__(self):
        return len(self.images_paths)
