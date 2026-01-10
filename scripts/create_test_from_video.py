import cv2 as cv
import random
import os
import shutil
from pathlib import Path
from cyclopts import App
app = App(name="Define Config for generating frames for testing:")


@app.command(name="download")
def download_frames_from_video(video_path:str, download_frames_path:str="data/extracted_frames", num_frames_per_sec=2, ):
    if os.path.exists(download_frames_path):
        shutil.rmtree(download_frames_path)
    os.makedirs(download_frames_path, exist_ok=True)
    cap=cv.VideoCapture(video_path)
    num_frames=int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps=int(cap.get(cv.CAP_PROP_FPS))
    duration=num_frames/fps
    print(f"Duration is {duration}s with fps={fps}. Generating {num_frames_per_sec} per sec")
    frame_ids=sorted([sec*fps+elem for sec in range(int(duration)) for elem in random.sample(range(int(fps)), num_frames_per_sec)])
    frame_counter=0
    saved_counter=0
    while True:
        if(saved_counter==len(frame_ids)):
            break
        ret, frame=cap.read()
        if not ret:
            print(f"Can't recieve stream from {video_path}")
            break
        if(frame_counter!=frame_ids[saved_counter]):
            frame_counter+=1
            continue
        cv.imwrite(f"{download_frames_path}/{Path(video_path).stem}_{saved_counter}.png", frame)
        saved_counter+=1
        frame_counter+=1
    cap.release()
if __name__=="__main__":
    app()
