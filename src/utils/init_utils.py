import torch
import random
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    This sets seeds for Python, NumPy and PyTorch (CPU and CUDA where available).
    It also configures a reasonable matmul precision for float32 operations.
    """
    torch.set_float32_matmul_precision("medium")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# def load_checkpoint(checkpoint_path):
#     """Load checkpoint from the specified path"""
#     if not os.path.exists(checkpoint_path):
#         raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
#     print(f"Loading checkpoint: {checkpoint_path}")
#     model = ModelFcosPretrained.load_from_checkpoint(checkpoint_path)
#     model.eval()
#     return model


# def resize_boxes(boxes, original_size, resized_size=800):
#     """
#     Resize bounding boxes from 800x800 back to original image size

#     Args:
#         boxes: tensor of shape (N, 4) with [x_min, y_min, x_max, y_max]
#         original_size: tuple of (height, width) original image size
#         resized_size: int, the size boxes were resized to (default 800)

#     Returns:
#         tensor of resized boxes
#     """
#     if len(boxes) == 0:
#         return boxes

#     orig_h, orig_w = original_size

#     # Calculate scale factors for height and width
#     scale_h = orig_h / resized_size
#     scale_w = orig_w / resized_size

#     # Scale boxes back
#     boxes_scaled = boxes.clone()
#     boxes_scaled[:, 0] = boxes[:, 0] * scale_w  # x_min
#     boxes_scaled[:, 1] = boxes[:, 1] * scale_h  # y_min
#     boxes_scaled[:, 2] = boxes[:, 2] * scale_w  # x_max
#     boxes_scaled[:, 3] = boxes[:, 3] * scale_h  # y_max

#     return boxes_scaled


# def draw_boxes(image, boxes, scores, labels, confidence_threshold=0.5):
#     """
#     Draw bounding boxes on image with scores and labels

#     Args:
#         image: numpy array of original image (BGR format for cv2)
#         boxes: tensor or numpy array of shape (N, 4) with [x_min, y_min, x_max, y_max]
#         scores: tensor or numpy array of scores
#         labels: tensor or numpy array of labels (0=background, 1=character)
#         confidence_threshold: only draw boxes above this threshold

#     Returns:
#         image with drawn boxes
#     """
#     image_with_boxes = image.copy()

#     # Convert to numpy if needed
#     if hasattr(boxes, "cpu"):
#         boxes = boxes.cpu().numpy()
#     if hasattr(scores, "cpu"):
#         scores = scores.cpu().numpy()
#     if hasattr(labels, "cpu"):
#         labels = labels.cpu().numpy()

#     # Draw each box
#     for i, (box, score) in enumerate(zip(boxes, scores)):
#         if score < confidence_threshold:
#             continue

#         x_min, y_min, x_max, y_max = map(int, box)

#         # Clip to image boundaries
#         x_min = max(0, x_min)
#         y_min = max(0, y_min)
#         x_max = min(image_with_boxes.shape[1], x_max)
#         y_max = min(image_with_boxes.shape[0], y_max)

#         # Draw rectangle (BGR format for cv2)
#         color = (0, 255, 0)  # Green for detected objects
#         cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), color, 2)

#         # Draw label and score
#         label_text = f"Char: {score:.2f}"
#         cv2.putText(
#             image_with_boxes,
#             label_text,
#             (x_min, y_min - 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             color,
#             2,
#         )

#     return image_with_boxes
