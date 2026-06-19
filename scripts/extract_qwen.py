import csv
import random
import shutil
from pathlib import Path

import pandas as pd
from cyclopts import App
from tqdm.auto import tqdm

from src.utils import delete_img_in_folder

app = App(name="Extract Qwen annotations")


@app.command(name="extract_qwen")
def extract_qwen(
    input_folder: str = "data/extracted_frames",
    dest_folder: str = "data/image_train_data",
    csv_file_path: str = "data/finshed_inference_qwen.csv",
    val_fraction: float = 0.2,
    seed: int = 42,
):
    """
    Build train/val folders from Qwen inference results.

    Reads bounding boxes from a CSV produced by scripts.inference_qwen, keeps only
    frames with game_state == "running", splits images into train/val, and writes
    images.csv in the same format as src.data_module.generate.
    """
    input_folder = Path(input_folder)
    dest_folder = Path(dest_folder)
    csv_path = Path(csv_file_path)
    df = pd.read_csv(csv_path)
    df = df[df["game_state"] == "running"]

    images: dict[str, list[tuple]] = {}
    file_to_videos: dict[str, str] = {}
    for _, row in df.iterrows():
        file_name = row["file_name"]
        images.setdefault(file_name, []).append(
            (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
                row["figure_color"],
            )
        )
        file_to_videos[file_name] = row["video_name"]

    file_names = sorted(images)
    random.seed(seed)
    random.shuffle(file_names)
    val_count = int(len(file_names) * val_fraction)
    splits = {
        "train": set(file_names[val_count:]),
        "val": set(file_names[:val_count]),
    }

    for split_name, split_files in splits.items():
        split_folder = dest_folder / split_name
        images_dir = split_folder / "images"
        delete_img_in_folder(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        csv_file = split_folder / "images.csv"

        with open(csv_file, "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(
                ["filename", "x_min", "y_min", "x_max", "y_max", "figure_color"]
            )
            for file_name in tqdm(sorted(split_files), desc=f"Writing {split_name}"):
                src = input_folder / f"{file_to_videos[file_name]}_{file_name}"
                if not src.exists():
                    continue
                shutil.copy2(src, images_dir / file_name)
                for x_min, y_min, x_max, y_max, figure_color in images[file_name]:
                    writer.writerow(
                        [file_name, x_min, y_min, x_max, y_max, figure_color]
                    )


if __name__ == "__main__":
    app()
