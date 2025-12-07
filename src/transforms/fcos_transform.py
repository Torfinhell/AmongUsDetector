import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class FcosTransform:
    """
    Basic FCOS transform that resizes images to a consistent size and converts to tensor.
    FCOS works best with images resized to multiples of 32 for FPN alignment.
    """

    def __init__(self, image_size=800, p=1.0):
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
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",  # Format: [x_min, y_min, x_max, y_max]
                label_fields=["class_labels"],
                min_area=0,
                min_visibility=0,
            ),
        )
        self.p = p

    def __call__(self, image, bboxes, class_labels=None):
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
        import random

        # Default to class label 0 if not provided
        if class_labels is None:
            class_labels = [0] * len(bboxes)

        # If no bboxes, skip albumentations (it can't handle empty bboxes)
        if len(bboxes) == 0:
            minimal_transform = A.Compose(
                [
                    A.Resize(
                        height=self.transform.transforms[0].height,
                        width=self.transform.transforms[0].width,
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ]
            )
            transformed = minimal_transform(image=image)
            return transformed["image"], bboxes, class_labels

        # Check probability
        if random.random() > self.p:
            # Return without transform if probability check fails
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
            transformed = minimal_transform(image=image)
            return transformed["image"], bboxes, class_labels

        # Apply full transform with bboxes
        transformed = self.transform(
            image=image, bboxes=bboxes, class_labels=class_labels
        )

        return transformed["image"], transformed["bboxes"], transformed["class_labels"]
