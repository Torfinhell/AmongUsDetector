import glob
import os
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import cv2 as cv
import easyocr
import numpy as np
from cyclopts import App
from PIL import Image

app = App(name="Extracting texts")
reader = easyocr.Reader(["en"])


@app.command(name="extract_text")
def extract_texts(image: str | np.ndarray):
    if isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))
    # image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image  = cv2.adaptiveThreshold(
    #     image,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     11,
    #     2
    # )
    texts = reader.readtext(image, detail=0)
    return texts


if __name__ == "__main__":
    app()
