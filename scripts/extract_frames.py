import glob
import os
import random
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import cv2 as cv
import editdistance
import faiss
import numpy as np
import pandas as pd
from cyclopts import App
from deepface import DeepFace
from tqdm.auto import tqdm

from scripts.create_face_db import ALIGN, FACE_THR, DetectorBackend, FaceEmbModel
from scripts.extract_texts import extract_texts
from src.utils import ALL_VIDEOS_PATHS, CsvChunkDownloader, delete_img_in_folder

app = App(name="Define Config for generating frames for testing:")
VIDEO_FOLDER = Path("data/videos")
EXCLUDE_VIDEOS = [
    str(VIDEO_FOLDER / "DUMBEST SIDEMEN AMONG US EVER.mp4")
]  # this is our test set
ALL_VIDEOS = [
    video_path for video_path in ALL_VIDEOS_PATHS if video_path not in EXCLUDE_VIDEOS
]
MODEL = FaceEmbModel.VGG_Face.value
USED_DETECTOR = DetectorBackend.FASTMTCNN.value
SIM_THRESHOLD = 0.60


@app.command(name="download")
def download_frames_from_video(
    video_folder: list[str] = ALL_VIDEOS,
    upload_frames_path: str = "data/extracted_frames",
    num_frames_per_sec: Optional[int] = 1,
    face_db="data/df_faces_final",
    filter_faces: bool = False,
    filter_text: bool = False,
    yandex_token: Optional[str] = None,
):
    upload_frames_path = Path(upload_frames_path)
    delete_img_in_folder(upload_frames_path)
    os.makedirs(upload_frames_path / "images", exist_ok=True)
    columns = ["file_name", "video_name"]
    if filter_text:
        columns.append("extracted_text")
        columns.append("game_state")
        columns.append("is_imposter")
    if filter_faces:
        columns.append("face_id")
        embeddings = []
        labels = []

        for img_path in list(Path(face_db).rglob("*.png")):
            rep = DeepFace.represent(
                img_path=str(img_path), model_name=MODEL, enforce_detection=False
            )
            embedding = np.array(rep[0]["embedding"], dtype=np.float32)
            embedding /= np.linalg.norm(embedding)
            embeddings.append(embedding)
            labels.append(img_path.parent.stem)
        embeddings = np.stack(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
    saved_counter = 0
    with CsvChunkDownloader(
        upload_frames_path / "images.csv",
        columns=columns,
        yandex_token=yandex_token,
        chunk_rows=1,
    ) as csv_download:
        for video_path in tqdm(video_folder, leave=True):
            print(video_path)
            cap = cv.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Can't open video {video_path}")
                continue
            num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv.CAP_PROP_FPS))
            if fps == 0 or num_frames == 0:
                print(f"Invalid video {video_path}: fps={fps}, frames={num_frames}")
                continue
            duration = num_frames / fps
            num_frames_per_sec = num_frames_per_sec or fps
            print(
                f"Duration is {duration}s with fps={fps}. Generating {num_frames_per_sec} per sec"
            )
            frame_ids = sorted(
                [
                    sec * fps + elem
                    for sec in range(int(duration))
                    for elem in random.sample(range(int(fps)), num_frames_per_sec)
                ]
            )
            frame_cnt = 0
            frames_chosen_cnt = 0
            for _ in tqdm(range(int(fps * duration)), leave=True):
                if frames_chosen_cnt == len(frame_ids):
                    break
                ret, frame = cap.read()
                if not ret:
                    print(f"Can't recieve stream from {video_path}")
                    break
                if frame_cnt != frame_ids[frames_chosen_cnt]:
                    frame_cnt += 1
                    continue
                file_name = f"{saved_counter}.png"
                data_row = []
                data_row.append(Path(file_name).name)
                data_row.append(Path(video_path).stem)
                if filter_text:
                    texts = extract_texts(frame)
                    data_row.append(texts)
                    res_filter = filter_text(texts)
                    if res_filter is not None:
                        data_row.append(res_filter)
                    else:
                        frame_cnt += 1
                        frames_chosen_cnt += 1
                        continue
                    data_row.append(
                        any(
                            [
                                has_word_inside(text, ["SADOUAGE", "SABOTAGE"])
                                for text in texts
                            ]
                        )
                    )
                if filter_faces:
                    face_detections = DeepFace.extract_faces(
                        img_path=frame,
                        detector_backend=USED_DETECTOR,
                        align=ALIGN,
                        enforce_detection=False,
                    )
                    matched_face_ids = []
                    for detection in face_detections:
                        if detection["confidence"] < FACE_THR:
                            continue
                        x, y, w, h = (
                            int(detection["facial_area"][key_x])
                            for key_x in ["x", "y", "w", "h"]
                        )
                        max_y, max_x, _ = frame.shape
                        xmin, ymin, xmax, ymax = (
                            max(0, x - w // 4),
                            max(0, y - h // 4),
                            min(x + w + w // 4, max_x),
                            min(y + h + h // 4, max_y),
                        )
                        cropped = cv.resize(frame[ymin:ymax, xmin:xmax], (400, 400))
                        query_emb = DeepFace.represent(
                            img_path=cropped, model_name=MODEL, enforce_detection=False
                        )[0]["embedding"]
                        query_emb = np.array(query_emb, dtype=np.float32)
                        query_emb /= np.linalg.norm(query_emb)
                        query_emb = query_emb.reshape(1, -1)
                        found_dist, found_index = index.search(query_emb, k=1)
                        if found_dist[0][0] > SIM_THRESHOLD:
                            matched_face_ids.append(labels[found_index[0][0]])
                    face_names = Counter(matched_face_ids)
                    if len(face_names) != 1:
                        continue
                    data_row.append(list(face_names)[0])
                csv_download.update_csv(pd.Series(data_row, index=columns))
                cv.imwrite(
                    f"{upload_frames_path}/images/{saved_counter}.png",
                    frame,
                )
                saved_counter += 1
                frames_chosen_cnt += 1
                frame_cnt += 1
            cap.release()


def filter_texts(texts):
    if any(
        [
            has_word_inside(
                text,
                [
                    "voting end",
                    "Voting Result",
                    "has voted",
                    "SKIP VOTE",
                    "Voting Result",
                    "VotingEnds",
                    "Proceeding In:",
                ],
                3,
            )
            for text in texts
        ]
    ):
        return "voting"
    elif any([has_word_inside(text, ["Who is t", "Begins In"], 3) for text in texts]):
        return "meeting"
    elif any([has_word_inside(text, ["Your role"], 3) for text in texts]):
        return "revealing role"
    elif any([has_word_inside(text, ["Fix Lights"], 3) for text in texts]):
        return "lights"
    elif any([has_word_inside(text, ["EMERGENCY"]) for text in texts]):
        return "emergency button"
    elif any([has_word_inside(text, ["dead body"]) for text in texts]):
        return "dead body"
    elif any([has_word_inside(text, ["Oxygen Depleted"], 3) for text in texts]):
        return "oxygen"
    elif any([has_word_inside(text, ["Reactor"], 3) for text in texts]):
        return "reactor"
    elif any([has_word_inside(text, ["ping: ms"], 4) for text in texts]):
        return "running"
    return None


def has_word_inside(text, words, num_subst=2):
    text = text.lower()
    words = [word.lower() for word in words]
    for word in words:
        wlen = len(word)
        if wlen > len(text):
            continue
        for i in range(len(text) - wlen + 1):
            window = text[i : i + wlen]
            if editdistance.eval(window, word) <= num_subst:
                return True

    return False


if __name__ == "__main__":
    app()
