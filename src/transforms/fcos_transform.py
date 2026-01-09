import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class FcosTransform: #TODO Use it for my implementation of model later
    """
    Basic FCOS transform that resizes images to a consistent size and converts to tensor.
    FCOS works best with images resized to multiples of 32 for FPN alignment.
    """

    def __init__(self, image_size=800):
        """
        Args:
            image_size: Target image size (will be resized to image_size x image_size)
            p: Probability of applying the transform (default 1.0 for always apply)
        """
        self.transform = A.Compose(
            [
                A.Resize(
                    height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR
                ),
                # A.Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225],
                #     max_pixel_value=255.0,
                # ),
                # #TODO dont need normalization, maybe later
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
        # If no bboxes, skip albumentations (it can't handle empty bboxes)
        if len(bboxes) == 0:
            transformed = self.transform(image=image)
            return transformed["image"], bboxes
        # Apply full transform with bboxes
        transformed = self.transform(image=image, bboxes=bboxes)
        return transformed["image"], transformed["bboxes"]
