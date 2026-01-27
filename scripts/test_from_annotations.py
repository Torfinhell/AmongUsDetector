import os
import xml.etree.ElementTree as ET
import csv
from pathlib import Path
import cv2
from cyclopts import App
app=App(name="Define Config create data from annotations")

@app.command(name="create")
def create_from_xml(annotations_dir:str, image_dir:str, output_folder:str):
    output_folder=Path(output_folder)
    (output_folder / "images").mkdir(exist_ok=True)
    image_dir=Path(image_dir)
    rows = []
    widths=[]
    heights=[]
    for file in os.listdir(annotations_dir):
        if not file.endswith(".xml"):
            continue

        xml_path = os.path.join(annotations_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        image=cv2.imread(image_dir/filename)
        height, width, _=image.shape
        cv2.imwrite(output_folder /"images" /filename, image)

        for obj in root.findall("object"):
            figure_color = obj.find("name").text

            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            assert 0<=xmin<=xmax<=width and 0<=ymin<=ymax<=height
            widths.append(xmax-xmin)
            heights.append(ymax-ymin)
            rows.append([
                filename,
                xmin,
                ymin,
                xmax,
                ymax,
                figure_color.lower()
            ])

    # Write CSV
    with open(output_folder /"images.csv", "w") as f:
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
if __name__=="__main__":
    app()

