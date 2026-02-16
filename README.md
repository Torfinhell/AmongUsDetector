# AmonUsDetector
I am implementing a among us detector based on AmongUs Sideman Series. In this readme i present two ways to train [Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) model. The first method uses synthesized data and can be later finetuned on real data. The second uses a dataset created using Qwen3-Vl 8B(about 20k images). Before that it was carefully filtered. As in checked what stage of the game the current frame is by extracting text with [EasyOCR](https://github.com/JaidedAI/EasyOCR). Also every frame was given a face_id for the face from a pre-collected database(using [faiss](https://github.com/facebookresearch/faiss)). The facemodel that extracted embeddings was VGGFace and BackendDetector was fastmtcnn(the library used was [DeepFace](https://github.com/serengil/deepface)). So in each section there can be two subsection for each way of training. The model was tested on data that was annotated on the CVAT platform.
Also yadisk logic was implemented for downloading csv files to yandex disk(see how to authorize [here](https://yandex.ru/dev/disk-api/doc/ru/concepts/quickstart#quickstart__oauth)), also used to synchronise inferencing Qwen-VL from different local machines(much faster)
## Results
TODO
## Installation

To create env and install dependecies:
```
conda create -n among_us python=3.11
conda activate among_us
pip install uv
uv sync
```
## Scripts
### For all
This script downloads checkpoint for inferencing, xml anotations for test, database face_db_final with people and their faces extracted from videos, extracted_frames for test and 4 maps from the among us game
```
uv run scripts/gdrive_download.py
```
To create test folder for training from xml anotations run:
```
uv run scripts/test_from_annotations.py create \
  --annotations_dir=data/annotations/Annotations/extracted_frames \
  --image_dir=data/extracted_frames \
  --output_folder=data/image_train_data/test

```
### To create Qwen3-Vl-annotated data
To create filtered data for Qwen3-Vl:
```
uv run -m scripts.extract_frames download \
 --upload_frames_path=data/extracted_frames_final \
 --filter_text=True \
 --filter_faces=True
Use argument --filter_faces only if face_db_final is created
Optional param:
--num_frames_per_sec(default: None)
--yandex_token=YANDEX_TOKEN(default: None)-specify this token if you want the csv to download to yadisk
```
Inference Qwen3-Vl on created dataset
```
uv run -m scripts.inference_qwen inference_qwen \
  --input_folder=data/extracted_frames_final \
  --output_folder=data/image_train_data/train \
  --yandex_token=YOUR_TOKEN
```
## Training and inference
### For synthesized data
To train model run:
```
 uv run train.py train_fcos \
  --val_folder=data/image_train_data/val \
  --train_folder=data/image_train_data/train \
  --test_folder=data/image_train_data/test \
  --batch_size=35 --num_epochs=300 \
  --seed=42 --val_epoch_len=50 \
  --train_epoch_len=500 \
  --swa_epoch_start=250 --background_folder=data/maps \
  --head_lr=8e-3 --backbone_lr=8e-3 \
  --swa_lrs=5e-4
Optional params(see src/configs for additinal training, model, datacreation or dataset params):
```
To inference model:
```
uv run inference.py run_inference \
  --checkpoint=checkpoints/last.ckpt \
  --pred_data=data/image_train_data/val/images\
  --pred_output=output
Optional params(see src/configs/all_configs.py for model or dataset configs)
```
### For Qwen-annotated data
To train model run:
```
 uv run train.py train_fcos \
  --val_folder=data/image_train_data/val \
  --train_folder=data/image_train_data/train \
  --test_folder=data/image_train_data/test \
  --batch_size=35 --num_epochs=300 \
  --seed=42 --val_epoch_len=50 \
  --train_epoch_len=500 --val_epoch_len=50 \
  --swa_epoch_start=250 --background_folder=data/maps \
  --head_lr=8e-3 --backbone_lr=8e-3 --swa_lrs=5e-4
Optional params(see src/configs for additinal training, model, datacreation or dataset params):
```
To inference model:
```
uv run inference.py run_inference \
  --checkpoint=checkpoints/last.ckpt \
  --pred_data=data/image_train_data/val/images\
  --pred_output=output
Optional params(see src/configs/all_configs.py for model or dataset configs)
```
## Inference Checkpoint on video
TODO
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
  --video_folder="data/videos/DUMBEST SIDEMEN AMONG US EVER.mp4" \
  --download_frames_path=data/extracted_frames

Optional param:
--num_frames_per_sec(default: None)

```
To generate dataset with synthesized figures:
```
uv run -m src.data_module.generate generate-data \
  --dest_folder=data/example \
  --num_generation=500 \
  --num_figures=20 \
  --augment_figure=True \
  --augment_mask=True \
  --draw_bbox=False \
  --background_folder=data/maps
```
To create face database(after that group by folders to produce face_db_final):
```
uv run -m scripts.create_face_db create_db_faces
Optinal param: num_frames_per_sec(default: 1),database_path(default: "data/db_faces")
```
To extract text from image_file using EasyOCR run:
```
 uv run -m scripts.extract_texts extract_text  --image=IMAGE_PATH
```
