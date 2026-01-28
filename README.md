# AmonUsDetector
I am implementing a among us detector based on AmongUs Sideman Series
## Installation

To create env and install dependecies:
```
conda create -n among_us python=3.11
conda activate among_us
pip install uv
uv sync
```
## Scripts

To download checkpoints and initialize xml annotations and checkpoints(all links are available at GDRIVE_URLS in this file) run:
```
uv run scripts/download_xml_ckpt.py 
```
To create test folder for training from xml anotations run:
```
uv run scripts/test_from_annotations.py create \
  --annotations_dir=data/annotations/Annotations \
  --image_dir=data/extracted_frames \
  --output_folder=data/image_train_data/test

```
### Additional scripts
Use this script to download videos from youtube(uploads to data/videos):
```
uv run scripts/download_videos.py
```
Use this script to validate correctness of bboxes in test data:
```
uv run -m scripts.check_data check_data \
  --images_folder=data/image_train_data/test/images \
  --images_csv=data/image_train_data/test/images.csv \
  --output_folder=data/check_bboxes
```
To extract frames from video for annotation:
```
uv run scripts/extract_frames.py download \
  --video_path="data/videos/DUMBEST SIDEMEN AMONG US EVER.mp4" \
  --download_frames_path=data/extracted_frames

Optional param: 
--num_frames_per_sec(default 2)

```
To generate dataset with synthesized figures:
```
uv run -m src.data_module.generate generate-data \
  --dest_folder=data/example \
  --num_generation=500 \
  --num_figures=20 \
  --augment_figure=True \
  --augment_mask=True \
  --head_lr=8e-3 \
  --backbone_lr=8e-3 \
  --draw_bbox=False
Optional params:
--background_folder=data/backgrounds(if exists)
```
## Traning and inference

To train model run:
```
 uv run train.py train_fcos \
  --val_folder=data/image_train_data/val \
  --train_folder=data/image_train_data/train \
  --test_folder=data/image_train_data/test \
  --num_epochs=300 \
  --train_epoch_len=500 \
  --val_epoch_len=50 \
  --seed=1 \
  --swa_epoch_start=250 \
  --swa_lrs=5e-3
Optional params(see src/configs/all_configs.py for model, datacreation or dataset configs)
```
To inference model:
```
uv run inference.py run_inference \
  --checkpoint=checkpoints/last.ckpt \
  --pred_data=data/image_train_data/val/images\
  --pred_output=output
Optional params(see src/configs/all_configs.py for model or dataset configs)
```