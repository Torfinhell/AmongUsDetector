import pandas as pd
import cv2
from pathlib import Path
from collections import defaultdict
from src.data_module.utils import color_to_ind, colors, hex_to_bgr
from dataclasses import dataclass
from cyclopts import App
app = App(name="Define Config to check data")
@app.command(name="check_data")
def check_data(images_folder:str, images_csv:str, output_folder:str): 
    annotations=pd.read_csv(images_csv)
    images_folder=Path(images_folder)
    output_folder=Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    detections_dict=defaultdict(list)
    labels_dict=defaultdict(list)
    for _,(filename, *row) in annotations.iterrows():
        detections_dict[filename].append(row[:4])
        labels_dict[filename].append(row[4])
    for filename, detections in detections_dict.items():
        image=cv2.imread(images_folder / filename)
        labels=labels_dict[filename]
        for label, detection in zip(labels, detections):
            xmin, ymin, xmax, ymax=detection
            hex_color=colors[color_to_ind[label]][1]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                          color=hex_to_bgr(hex_color),thickness=2)
        cv2.imwrite(output_folder  /filename, image)
if __name__=="__main__":
    app()