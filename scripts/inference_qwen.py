from pathlib import Path
from typing import Optional

import pandas as pd
import supervision as sv
import torch
from cyclopts import App
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.utils import (
    ALL_VIDEOS_PATHS,
    CsvChunkDownloader,
    colors,
    distance_bgr,
    hex_to_bgr,
)

app = App(name="Define Config for generating frames for testing:")
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
UPDATE_CSV_EVERY_IMAGE = 400
VIDEO_FOLDER = Path("data/videos")
EXCLUDE_VIDEOS = [
    str(VIDEO_FOLDER / "DUMBEST SIDEMEN AMONG US EVER.mp4")
]  # this is our test set
ALL_VIDEOS = [
    Path(video_path)
    for video_path in ALL_VIDEOS_PATHS
    if video_path not in EXCLUDE_VIDEOS
]


@app.command(name="inference_qwen")
def inference_qwen(
    input_folder,
    output_folder: str = "",
    max_new_tokens: int = 1024,
    yandex_token: Optional[str] = None,
):
    """
    Docstring for inference_qwen

    :param input_folder: contains images.csv with info and images folder
    :param max_new_tokens: max tokens that qwen can output
    :type max_new_tokens: int
    :param output_folder: output folder containing the same images and images.csv
    :type output_folder: str
    :param yandex_token: token to download to yandex disk(see https://oauth.yandex.ru/). Will be used to inference multiple qwens on seperate accounts in kaggle
    :type output_folder: Optional[str]
    """
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    propmpt = """
    Detect all among us figures in the image.
    Return ONLY valid JSON in the following format:
    [
    {"label": "<class_name>", "bbox_2d": [x1, y1, x2, y2]}
    ]
    No extra text.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    input_csv = pd.read_csv(input_folder / "images.csv")
    csv_start = "started_inference_qwen.csv"
    csv_finish = "finshed_inference_qwen.csv"
    with CsvChunkDownloader(
        csv_start,
        columns=["video_name"],
        download_from_disk=True,
        yandex_token=yandex_token,
        chunk_rows=1,
    ) as started_csv:
        with CsvChunkDownloader(
            csv_finish,
            columns=list(input_csv.columns)
            + ["xmin", "ymin", "xmax", "ymax", "figure_color"],
            download_from_disk=True,
            yandex_token=yandex_token,
            chunk_rows=None,
        ) as finished_csv:
            while True:
                current_videos = set(
                    video_name
                    for video_name in started_csv.get_csv(
                        default_columns=["video_name"]
                    )["video_name"]
                ) | set(
                    video_name
                    for video_name in finished_csv.get_csv(
                        default_columns=["video_name"]
                    )["video_name"]
                )
                videos_left = set([file.stem for file in ALL_VIDEOS]) - current_videos
                if len(videos_left) == 0:
                    break
                video_name = list(videos_left)[0]
                started_csv.update_csv(pd.Series([video_name], index=["video_name"]))
                for idx, row in input_csv[
                    input_csv["video_name"] == video_name
                ].iterrows():
                    file_name = row["file_name"]
                    extracted_text = row["extracted_text"]
                    game_state = row["game_state"]
                    is_imposter = row["is_imposter"]
                    face_id = row["face_id"]
                    # if game_state != "running":
                    #     continue
                    image_path = input_folder / "images" / file_name
                    image = Image.open(image_path).convert("RGB")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": propmpt},
                            ],
                        }
                    ]
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to("cuda")
                    with torch.inference_mode():
                        gen = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                        )
                    trimmed = [g[len(i) :] for i, g in zip(inputs.input_ids, gen)]
                    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                    detections = sv.Detections.from_vlm(
                        vlm=sv.VLM.QWEN_3_VL, result=text, resolution_wh=image.size
                    )
                    for xmin, ymin, xmax, ymax in detections.xyxy:
                        xmin, ymin, xmax, ymax = (
                            int(xmin),
                            int(ymin),
                            int(xmax),
                            int(ymax),
                        )
                        color_name = get_figure_color(image[xmin:xmax, ymin:ymax, :])
                        finished_csv.update_csv(
                            pd.Series(
                                [
                                    file_name,
                                    video_name,
                                    extracted_text,
                                    game_state,
                                    is_imposter,
                                    face_id,
                                    xmin,
                                    ymin,
                                    xmax,
                                    ymax,
                                    color_name,
                                ]
                            )
                        )


def get_figure_color(image):
    h, w, d = image.shape
    assert d == 3
    center_pixel = (image[h - h // 10 : h + h // 10, w - w // 10 : w + w // 10]).mean(
        dim=-1
    )
    min_dist = None
    ans_color = None
    for color_name, hex_color in colors:
        dist = distance_bgr(center_pixel, hex_to_bgr(hex_color))
        if min_dist is None or dist < min_dist:
            min_dist = dist
            ans_color = color_name
    return ans_color


if __name__ == "__main__":
    app()
