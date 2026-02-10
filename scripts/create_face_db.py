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

@app.command(name="create_db")
def register_faces(video_folder:Union[str, list[str]]=ALL_VIDEOS, download_frames_path:str="data/extracted_frames", num_frames_per_sec:Optional[int]=None, one_face:bool=False):
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
        saved_counter=0
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

            
    cap.release()
if __name__=="__main__":
    app()
