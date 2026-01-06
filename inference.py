import os
import cv2
from models.fcos_pretrained import ModelFcosPretrained
from cyclopts import App
from src.configs import ModelPredConfig
from src.utils import set_seed
from src.data_module import AmongUsDatamodule

app = App(name="Define Config for inferencing:")


def load_checkpoint(checkpoint_path):
    """Load checkpoint from the specified path"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading checkpoint: {checkpoint_path}")
    model = ModelFcosPretrained.load_from_checkpoint(checkpoint_path)
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


@app.command
def run_inference(cfg: ModelPredConfig):
    """
    Run inference on all images in a directory and save results with bounding boxes

    Args:
        image_dir: directory containing images to process
        output_dir: directory to save output images with bounding boxes
        checkpoint_path: path to the model checkpoint file
        confidence_threshold: confidence threshold for drawing boxes
    """
    training_cfg = cfg.training_cfg
    set_seed(training_cfg.seed)

    # initialize Datamodule
    data_module = AmongUsDatamodule(cfg.datamodule_cfg, cfg.creation_cfg)
    # Setup TensorBoard logger
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="loss_val",
        mode="min",
        save_top_k=3,
        dirpath="checkpoints",
        filename="fcos-{epoch:02d}-{loss_val:.4f}",
    )
    # Gradient Norm Output
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=training_cfg.num_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        gradient_clip_val=6.0,
        enable_progress_bar=True,
        limit_train_batches=training_cfg.train_epoch_len,
        limit_val_batches=training_cfg.val_epoch_len,
    )
    model = ModelFcosPretrained(cfg)
    trainer.fit(model=model, datamodule=data_module)
    return trainer, model


if __name__ == "__main__":
    app()
