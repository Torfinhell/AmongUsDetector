import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.configs import TransformConfig
import random
class FcosTransform:
    """
    Basic FCOS transform that resizes images to a consistent size and converts to tensor.
    FCOS works best with images resized to multiples of 32 for FPN alignment.
    """

    def __init__(self, transform_cfg:TransformConfig, part="train"):
        """
        Args:
            image_size: Target image size (will be resized to image_size x image_size)
        """
        self.normalize=transform_cfg.normalize
        self.height_range=transform_cfg.height_range
        self.width_range=transform_cfg.width_range
        self.part=part
        if(part!="train"):
            self.update_transform(sum(self.height_range)/2, sum(self.width_range)/2)
    def update_transform(self, new_height, new_width):
        normalize_trans=A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ) if self.normalize else A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0) 
        self.transform = A.Compose(
            [
                A.Resize(new_height, new_width), #TODO maybe change later to A.RandomResizeCrop()
                normalize_trans,
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",  # Format: [x_min, y_min, x_max, y_max]
                min_area=0,
                min_visibility=0,
            ),
        )

    def __call__(self, image, bboxes):
        """
        Args:
            image: numpy array of shape (H, W, 3)
            bboxes: list of bounding boxes in format [x_min, y_min, x_max, y_max]
            class_labels: list of class labels for each bbox (default: 0 for all)

        Returns:
            image: torch tensor of shape (3, H, W) normalized
            bboxes: list of normalized bounding boxes
            class_labels: list of class labels
        """
        if(self.part=="train"):
            self.update_transform(random.randrange(*self.height_range), random.randrange(*self.width_range))
        # If no bboxes, skip albumentations (it can't handle empty bboxes)
        if len(bboxes) == 0:
            transformed = self.transform(image=image)
            return transformed["image"], bboxes
        # Apply full transform with bboxes
        transformed = self.transform(image=image, bboxes=bboxes)
        return transformed["image"], transformed["bboxes"]
