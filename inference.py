import os
import torch
import cv2
from glob import glob
from src.models.fcos import ModelFcos
from src.transforms.fcos_transform import FcosTransform


def load_checkpoint(checkpoint_dir="checkpoints"):
    """Load the latest checkpoint from the checkpoints directory"""
    checkpoint_files = sorted(glob(os.path.join(checkpoint_dir, "*.ckpt")))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Load the first checkpoint (or you could load the best one based on loss)
    checkpoint_path = checkpoint_files[0]
    print(f"Loading checkpoint: {checkpoint_path}")

    model = ModelFcos.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def resize_boxes(boxes, original_size, resized_size=800):
    """
    Resize bounding boxes from 800x800 back to original image size

    Args:
        boxes: tensor of shape (N, 4) with [x_min, y_min, x_max, y_max]
        original_size: tuple of (height, width) original image size
        resized_size: int, the size boxes were resized to (default 800)

    Returns:
        tensor of resized boxes
    """
    if len(boxes) == 0:
        return boxes

    orig_h, orig_w = original_size

    # Calculate scale factors for height and width
    scale_h = orig_h / resized_size
    scale_w = orig_w / resized_size

    # Scale boxes back
    boxes_scaled = boxes.clone()
    boxes_scaled[:, 0] = boxes[:, 0] * scale_w  # x_min
    boxes_scaled[:, 1] = boxes[:, 1] * scale_h  # y_min
    boxes_scaled[:, 2] = boxes[:, 2] * scale_w  # x_max
    boxes_scaled[:, 3] = boxes[:, 3] * scale_h  # y_max

    return boxes_scaled


def draw_boxes(image, boxes, scores, labels, confidence_threshold=0.5):
    """
    Draw bounding boxes on image with scores and labels

    Args:
        image: numpy array of original image (BGR format for cv2)
        boxes: tensor or numpy array of shape (N, 4) with [x_min, y_min, x_max, y_max]
        scores: tensor or numpy array of scores
        labels: tensor or numpy array of labels (0=background, 1=character)
        confidence_threshold: only draw boxes above this threshold

    Returns:
        image with drawn boxes
    """
    image_with_boxes = image.copy()

    # Convert to numpy if needed
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()

    # Draw each box
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score < confidence_threshold:
            continue

        x_min, y_min, x_max, y_max = map(int, box)

        # Clip to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_with_boxes.shape[1], x_max)
        y_max = min(image_with_boxes.shape[0], y_max)

        # Draw rectangle (BGR format for cv2)
        color = (0, 255, 0)  # Green for detected objects
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), color, 2)

        # Draw label and score
        label_text = f"Char: {score:.2f}"
        cv2.putText(
            image_with_boxes,
            label_text,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return image_with_boxes


def run_inference(
    image_dir="data/image_train/images",
    output_dir="output",
    checkpoint_dir="checkpoints",
    confidence_threshold=0.5,
):
    """
    Run inference on all images in a directory and save results with bounding boxes

    Args:
        image_dir: directory containing images to process
        output_dir: directory to save output images with bounding boxes
        checkpoint_dir: directory containing model checkpoints
        confidence_threshold: confidence threshold for drawing boxes
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_checkpoint(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create transform
    transform = FcosTransform(image_size=800, p=0.0)  # No augmentation for inference

    # Get all image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, ext)))
        image_files.extend(glob(os.path.join(image_dir, ext.upper())))

    image_files = sorted(set(image_files))  # Remove duplicates and sort

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images to process")
    print(f"Processing images from: {image_dir}")
    print(f"Saving results to: {output_dir}")

    # Process each image
    with torch.no_grad():
        for i, image_path in enumerate(image_files):
            try:
                # Load original image
                original_image = cv2.imread(image_path)
                if original_image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                original_size = original_image.shape[:2]  # (height, width)

                # Prepare image for model (resize to 800x800 and normalize)
                # Note: transform expects PIL or numpy image, let's use numpy directly
                image_np = cv2.cvtColor(
                    original_image, cv2.COLOR_BGR2RGB
                )  # Convert BGR to RGB for processing

                # Apply transform
                try:
                    # We need to apply transform manually since it expects bboxes
                    # For inference, we don't have ground truth boxes, so we'll resize manually
                    import albumentations as A

                    transform_pipeline = A.Compose(
                        [
                            A.Resize(height=800, width=800),
                            A.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ],
                        bbox_params=A.BboxParams(
                            format="pascal_voc", min_area=0, min_visibility=0
                        ),
                    )

                    # Apply transform without bboxes (we pass empty list)
                    transformed = transform_pipeline(image=image_np, bboxes=[])
                    image_transformed = transformed["image"]

                    # Convert to tensor and add batch dimension
                    image_tensor = (
                        torch.from_numpy(image_transformed).permute(2, 0, 1).float()
                    )
                    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

                except Exception as e:
                    print(f"Error transforming image {image_path}: {e}")
                    continue

                # Run inference
                image_tensor = image_tensor.to(device)
                predictions = model.model(
                    [image_tensor.squeeze(0)]
                )  # Remove batch dim for FCOS

                # Extract predictions
                boxes = predictions[0]["boxes"]  # (N, 4)
                scores = predictions[0]["scores"]  # (N,)
                labels = predictions[0]["labels"]  # (N,)

                # Resize boxes back to original image size
                boxes_original = resize_boxes(boxes, original_size, resized_size=800)

                # Draw boxes on original image
                image_with_boxes = draw_boxes(
                    original_image,
                    boxes_original,
                    scores,
                    labels,
                    confidence_threshold=confidence_threshold,
                )

                # Save result
                output_filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, image_with_boxes)

                print(
                    f"[{i+1}/{len(image_files)}] Processed: {image_path} -> {output_path}"
                )
                print(
                    f"  Detections: {len(scores)} boxes, {sum(scores > confidence_threshold)} above threshold"
                )

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

    print(f"\nInference complete! Results saved to {output_dir}")


if __name__ == "__main__":
    run_inference()
