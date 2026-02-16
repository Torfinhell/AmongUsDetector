import os
import random
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import cv2 as cv
from cyclopts import App
from deepface import DeepFace
from tqdm.auto import tqdm

from src.utils import ALL_VIDEOS_PATHS, delete_img_in_folder

app = App(name="Define Config for generating frames for testing:")
VIDEO_FOLDER = Path("data/videos")
EXCLUDE_VIDEOS = [
    str(VIDEO_FOLDER / "DUMBEST SIDEMEN AMONG US EVER.mp4")
]  # this is our test set
ALL_VIDEOS = [
    video_path for video_path in ALL_VIDEOS_PATHS if video_path not in EXCLUDE_VIDEOS
]


class FaceEmbModel(Enum):
    VGG_Face = "VGG-Face"
    Facenet = "Facenet"
    Facenet512 = "Facenet512"
    OpenFace = "OpenFace"
    DeepFace = "DeepFace"
    DeepID = "DeepID"
    ArcFace = "ArcFace"
    Dlib = "Dlib"
    SFace = "SFace"
    GhostFaceNet = "GhostFaceNet"
    Buffalo_L = "Buffalo_L"


MODEL = FaceEmbModel.VGG_Face.value


class FaceMetrics(Enum):
    COSINE = "cosine"
    EUCLIDIAN = "euclidean"
    EUCLIDIAN_L2 = "euclidean_l2"
    ANGULAR = "angular"


USED_METRIC = FaceMetrics.EUCLIDIAN_L2.value


class DetectorBackend(Enum):
    OPENCV = "opencv"
    SSD = "ssd"
    DLIB = "dlib"
    MTCNN = "mtcnn"
    FASTMTCNN = "fastmtcnn"
    RETINAFACE = "retinaface"
    MEDIAPIPE = "mediapipe"
    YOLO8N = "yolov8n"
    YOLO8M = "yolov8m"
    YOLO8L = "yolov8l"
    YOLO11N = "yolov11n"
    YOLO11S = "yolov11s"
    YOLO11M = "yolov11m"
    YOLO11L = "yolov11l"
    YOLO12N = "yolov12n"
    YOLO12S = "yolov12s"
    YOLO12M = "yolov12m"
    YOLO12L = "yolov12l"
    YUNET = "yunet"
    CENTERFACE = "centerface"


USED_DETECTOR = DetectorBackend.FASTMTCNN.value
# see https://github.com/serengil/deepface/tree/master/benchmarks
ALIGN = True
FACE_THR = 0.99


@app.command(name="create_db_faces")
def register_faces(
    video_folders: Union[str, list[str]] = ALL_VIDEOS,
    database_path: str = "data/db_faces",
    num_frames_per_sec: Optional[int] = 1,
):
    if not isinstance(video_folders, list):  # video folder can be path
        video_folders = [video_folders]
    delete_img_in_folder(database_path)
    os.makedirs(database_path, exist_ok=True)
    cnt_faces = 0
    for video_path in tqdm(video_folders, leave=True):
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
        chosen_frames_cnt = 0
        while Path(f"{database_path}/{cnt_faces}.png").exists():
            cnt_faces += 1
        for _ in tqdm(range(int(fps * duration)), leave=True):
            if chosen_frames_cnt == len(frame_ids):
                break
            ret, frame = cap.read()
            if not ret:
                print(f"Can't recieve stream from {video_path}")
                break
            if frame_cnt != frame_ids[chosen_frames_cnt]:
                frame_cnt += 1
                continue
            face_detections = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=USED_DETECTOR,
                align=ALIGN,
                enforce_detection=False,
            )
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
                if not len(os.listdir(database_path)):
                    cv.imwrite(f"{database_path}/{cnt_faces}.png", cropped)
                    cnt_faces += 1
                else:
                    dfs = DeepFace.find(
                        cropped,
                        db_path=database_path,
                        model_name=MODEL,
                        enforce_detection=False,
                        detector_backend=USED_DETECTOR,
                        distance_metric=USED_METRIC,
                        align=ALIGN,
                        silent=True,
                    )
                    if dfs[0].empty:
                        cv.imwrite(f"{database_path}/{cnt_faces}.png", cropped)
                        cnt_faces += 1
            frame_cnt += 1
            chosen_frames_cnt += 1
        cap.release()


if __name__ == "__main__":
    app()
