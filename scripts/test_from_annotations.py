import os
import xml.etree.ElementTree as ET
import csv
from pathlib import Path
import cv2
ANNOTATIONS_DIR = "data/annotations/Annotations"
IMAGE_DIR=Path("data/extracted_frames")
OUTPUT_FOLDER = Path("data/image_train_data/test")
(OUTPUT_FOLDER / "images").mkdir(exist_ok=True)

rows = []

for file in os.listdir(ANNOTATIONS_DIR):
    if not file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATIONS_DIR, file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    print(OUTPUT_FOLDER /"images" /filename)
    image=cv2.imread(IMAGE_DIR/filename)
    cv2.imwrite(OUTPUT_FOLDER /"images" /filename, image)
    for obj in root.findall("object"):
        figure_color = obj.find("name").text

        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        rows.append([
            filename,
            xmin,
            ymin,
            xmax,
            ymax,
            figure_color.lower()
        ])

# Write CSV
with open(OUTPUT_FOLDER /"images.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "figure_color"
    ])
    writer.writerows(rows)

