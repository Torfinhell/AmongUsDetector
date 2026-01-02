import os
import torch
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.fcos_pretrained import ModelFcosPretrained


# -------------------------------
# Dataset for video frames
# -------------------------------
class VideoFrameDataset(Dataset):
    """Dataset for loading and preprocessing video frames in batches"""

    def __init__(self, video_path, max_frames=None):
        self.video_path = video_path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if max_frames and max_frames < self.frame_count:
            self.frame_count = max_frames

        self.frame_indices = list(range(self.frame_count))

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        original_frame = frame.copy()

        # Preprocess
        frame_resized = cv2.resize(frame, (800, 800))
        frame_normalized = frame_resized.astype("float32") / 255.0
        frame_normalized = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2RGB)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        frame_normalized = (frame_normalized - mean) / std
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).float()

        return idx, frame_tensor, original_frame

    def get_video_info(self):
        return {
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


# -------------------------------
# Collate function for batching
# -------------------------------
def collate_fn_frames(batch):
    indices = [item[0] for item in batch]
    frames = [item[1] for item in batch]
    originals = [item[2] for item in batch]
    frames_batch = torch.stack(frames, dim=0)
    return indices, frames_batch, originals


# -------------------------------
# Load model checkpoint
# -------------------------------
def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = ModelFcosPretrained.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


# -------------------------------
# Resize bounding boxes
# -------------------------------
def resize_boxes(boxes, original_size, resized_size=800):
    if len(boxes) == 0:
        return boxes

    orig_h, orig_w = original_size
    scale_h = orig_h / resized_size
    scale_w = orig_w / resized_size

    boxes_scaled = boxes.clone()
    boxes_scaled[:, 0] = boxes[:, 0] * scale_w
    boxes_scaled[:, 1] = boxes[:, 1] * scale_h
    boxes_scaled[:, 2] = boxes[:, 2] * scale_w
    boxes_scaled[:, 3] = boxes[:, 3] * scale_h

    return boxes_scaled


# -------------------------------
# Draw bounding boxes
# -------------------------------
def draw_boxes(image, boxes, scores, labels, confidence_threshold=0.5):
    image_with_boxes = image.copy()

    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < confidence_threshold:
            continue

        x_min, y_min, x_max, y_max = map(int, box)

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_with_boxes.shape[1], x_max)
        y_max = min(image_with_boxes.shape[0], y_max)

        color = (0, 255, 0)
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), color, 2)

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


# -------------------------------
# Process batch
# -------------------------------
def process_batch(model, batch_frames, device, original_sizes):
    batch_frames = batch_frames.to(device)
    all_predictions = []

    for i in range(batch_frames.shape[0]):
        frame_tensor = batch_frames[i]
        predictions = model.model([frame_tensor])

        boxes = predictions[0]["boxes"]
        scores = predictions[0]["scores"]
        labels = predictions[0]["labels"]

        boxes_original = resize_boxes(boxes, original_sizes[i], resized_size=800)

        all_predictions.append(
            {"boxes": boxes_original, "scores": scores, "labels": labels}
        )

    return all_predictions


# -------------------------------
# Run video inference with batching
# -------------------------------
def run_video_inference_batch(
    video_path,
    output_path,
    checkpoint_path,
    confidence_threshold=0.5,
    batch_size=8,
    max_frames=None,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model = load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}, batch size: {batch_size}")

    dataset = VideoFrameDataset(video_path, max_frames=max_frames)
    video_info = dataset.get_video_info()

    print(
        f"Video info: {video_info['frame_count']} frames, {video_info['fps']:.2f} fps, "
        f"{video_info['width']}x{video_info['height']} resolution"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_frames,
        num_workers=0,
    )  # num_workers=0!

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        video_info["fps"],
        (video_info["width"], video_info["height"]),
    )

    if not out.isOpened():
        raise IOError(f"Cannot create output video: {output_path}")

    processed_frames = {}

    with torch.no_grad():
        for batch_indices, batch_frames, batch_originals in tqdm(
            dataloader, desc="Processing batches"
        ):
            original_sizes = [frame.shape[:2] for frame in batch_originals]
            batch_predictions = process_batch(
                model, batch_frames, device, original_sizes
            )

            for idx, original_frame, predictions in zip(
                batch_indices, batch_originals, batch_predictions
            ):
                frame_with_boxes = draw_boxes(
                    original_frame,
                    predictions["boxes"],
                    predictions["scores"],
                    predictions["labels"],
                    confidence_threshold,
                )
                processed_frames[idx] = frame_with_boxes

    # Write frames in order
    print("\nWriting frames to output video...")
    for i in tqdm(range(len(dataset)), desc="Writing frames"):
        if i in processed_frames:
            frame = processed_frames[i]
        else:
            # Write black frame if frame wasn't processed
            frame = np.zeros(
                (video_info["height"], video_info["width"], 3), dtype=np.uint8
            )
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()
    print(f"\nVideo inference complete! Saved to {output_path}")


# -------------------------------
# Main CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch FCOS video inference")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--max_frames", type=int, default=None, help="Maximum frames to process"
    )

    args = parser.parse_args()

    run_video_inference_batch(
        args.input,
        args.output,
        args.checkpoint,
        confidence_threshold=args.confidence,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
    )
