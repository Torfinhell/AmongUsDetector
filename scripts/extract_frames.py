import cv2 as cv
import random
import os
import shutil
from pathlib import Path
from typing import Optional, Union
from pathlib import Path
from cyclopts import App
from tqdm.auto import tqdm
from deepface import DeepFace
app = App(name="Define Config for generating frames for testing:")
VIDEO_FOLDER=Path("data/videos")
EXCLUDE_VIDEOS=[VIDEO_FOLDER/"DUMBEST SIDEMEN AMONG US EVER.mp4"]
ALL_VIDEOS=[video_path for video_path in VIDEO_FOLDER.iterdir() if video_path not in EXCLUDE_VIDEOS]
metrics = ["cosine", "euclidean", "euclidean_l2", "angular"]
USED_METRIC=metrics[0]
backends = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8n', 'yolov8m', 
    'yolov8l', 'yolov11n', 'yolov11s', 'yolov11m',
    'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m',
    'yolov12l', 'yunet', 'centerface',
]
USED_DETECTOR = backends[3]
ALIGN = True
@app.command(name="download")#filter with faces and then filter with text
def download_frames_from_video(video_folder:Union[str, list[str]]=ALL_VIDEOS, download_frames_path:str="data/extracted_frames", num_frames_per_sec:Optional[int]=None, one_face:bool=False):
    if not isinstance(video_folder, list):#video folder can be path
        video_folder=[video_folder]
    if os.path.exists(download_frames_path):
        shutil.rmtree(download_frames_path)
    os.makedirs(download_frames_path, exist_ok=True)
    for video_path in tqdm(video_folder, leave=True):
        print(video_path)
        cap=cv.VideoCapture(video_path)
        num_frames=int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps=int(cap.get(cv.CAP_PROP_FPS))
        duration=num_frames/fps
        num_frames_per_sec=num_frames_per_sec or fps
        print(f"Duration is {duration}s with fps={fps}. Generating {num_frames_per_sec} per sec")
        frame_ids=sorted([sec*fps+elem for sec in range(int(duration)) for elem in random.sample(range(int(fps)), num_frames_per_sec)])
        frame_counter=0
        sample_counter=0
        saved_counter=0
        sample_id=0
        prev_face=None
        for _ in tqdm(range(int(fps*duration)), leave=True):
            if(saved_counter==len(frame_ids)):
                break
            ret, frame=cap.read()
            if not ret:
                print(f"Can't recieve stream from {video_path}")
                break
            if(frame_counter!=frame_ids[saved_counter]):
                frame_counter+=1
                continue
            if(one_face):
                faces=get_faces(frame)
                if(len(faces)!=1):
                    if(prev_face is not None):
                        sample_id+=1
                        sample_counter=0
                    prev_face=None
                    continue
                if(prev_face is None or similiar_faces(prev_face, faces[0])):
                    cv.imwrite(f"{download_frames_path}/{Path(video_path).stem}/{sample_id}/{sample_counter}.png", frame)
                    sample_counter+=1
                else:
                    sample_id+=1
                    sample_counter=0
                    cv.imwrite(f"{download_frames_path}/{Path(video_path).stem}/{sample_id}/{sample_counter}.png", frame)
            else:
                cv.imwrite(f"{download_frames_path}/{Path(video_path).stem}/{saved_counter}.png", frame)
            saved_counter+=1
            frame_counter+=1
    cap.release()
def get_faces(frame):
    face_objs = DeepFace.extract_faces(frame)
    return [frame]
def similiar_faces(face1, face2):
    retdistance=DeepFace.verify(
        img1_path = "img1.jpg", img2_path = "img2.jpg", distance_metric = metrics[1]
    )
if __name__=="__main__":
    app()
