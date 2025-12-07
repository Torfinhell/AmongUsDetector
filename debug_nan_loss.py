#!/usr/bin/env python
"""
Debug script to diagnose NaN loss issues in FCOS training.
Run this to identify problematic data.
"""
import torch
from src.datasets import AmongUsImagesDataset
from src.transforms import FcosTransform
from torch.utils.data import DataLoader
from train import collate_fn

print("=" * 80)
print("DEBUGGING NAN LOSS IN FCOS TRAINING")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
try:
    dataset = AmongUsImagesDataset(
        "data/image_train", transform=FcosTransform(image_size=800)
    )
    print(f"✓ Dataset loaded: {len(dataset)} images")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    exit(1)

# Check individual samples
print("\n2. Checking individual samples...")
num_samples = min(5, len(dataset))
for i in range(num_samples):
    try:
        image, target = dataset[i]
        boxes = target["boxes"]
        labels = target["labels"]

        print(f"\n  Sample {i}:")
        print(f"    Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"    Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"    Has NaN in image: {torch.isnan(image).any()}")
        print(f"    Boxes shape: {boxes.shape}")

        if len(boxes) > 0:
            print(
                f"    Boxes range - x: [{boxes[:, 0].min():.1f}, {boxes[:, 2].max():.1f}]"
            )
            print(
                f"    Boxes range - y: [{boxes[:, 1].min():.1f}, {boxes[:, 3].max():.1f}]"
            )

            # Check for invalid boxes
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            print(f"    Width range: [{width.min():.1f}, {width.max():.1f}]")
            print(f"    Height range: [{height.min():.1f}, {height.max():.1f}]")

            # Check for NaN
            if torch.isnan(boxes).any():
                print(f"    ⚠️  NaN in boxes!")
                print(f"    {boxes}")

            # Check for invalid boxes (negative dimensions)
            invalid = (width <= 0) | (height <= 0)
            if invalid.any():
                print(f"    ⚠️  Invalid boxes (width/height <= 0)!")
                print(f"    Invalid count: {invalid.sum()}/{len(boxes)}")
        else:
            print(f"    No boxes in this image")

        print(f"    Labels: {labels}")

    except Exception as e:
        print(f"  ✗ Error processing sample {i}: {e}")

# Test DataLoader with batching
print("\n3. Testing DataLoader with batching...")
try:
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=0)

    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx > 2:  # Only check first 3 batches
            break

        print(f"\n  Batch {batch_idx}:")
        print(f"    Num images: {len(images)}")
        print(f"    Num targets: {len(targets)}")

        for i, (img, target) in enumerate(zip(images, targets)):
            print(f"\n    Image {i} in batch:")
            print(f"      Shape: {img.shape}")
            print(f"      Has NaN: {torch.isnan(img).any()}")
            print(f"      Has Inf: {torch.isinf(img).any()}")
            print(f"      Boxes shape: {target['boxes'].shape}")

            if torch.isnan(target["boxes"]).any():
                print(f"      ⚠️  NaN in boxes!")

            if torch.isinf(target["boxes"]).any():
                print(f"      ⚠️  Inf in boxes!")

        # Test forward pass through model
        print(f"\n  Testing model forward pass on batch {batch_idx}...")
        try:
            from src.models import ModelFcos

            model = ModelFcos()
            model.eval()

            with torch.no_grad():
                # Move to GPU if available
                if torch.cuda.is_available():
                    images_cuda = [img.cuda() for img in images]
                    targets_cuda = [
                        {k: v.cuda() if torch.is_tensor(v) else v for k, v in t.items()}
                        for t in targets
                    ]
                else:
                    images_cuda = images
                    targets_cuda = targets

                # Forward pass (training mode to test loss computation)
                model.model.train()
                loss_dict = model.model(images_cuda, targets_cuda)

                print(f"    Loss dict keys: {loss_dict.keys()}")
                for key, value in loss_dict.items():
                    print(f"      {key}: {value.item():.6f}")
                    if torch.isnan(value):
                        print(f"      ⚠️  NaN in {key}!")
                    if torch.isinf(value):
                        print(f"      ⚠️  Inf in {key}!")

                total_loss = sum(
                    v for v in loss_dict.values() if isinstance(v, torch.Tensor)
                )
                print(f"    Total loss: {total_loss.item():.6f}")
                if torch.isnan(total_loss):
                    print(f"    ⚠️  TOTAL LOSS IS NAN!")

        except Exception as e:
            print(f"    ✗ Error in forward pass: {e}")
            import traceback

            traceback.print_exc()

except Exception as e:
    print(f"✗ Error in DataLoader: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
