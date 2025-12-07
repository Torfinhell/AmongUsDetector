import cv2
import numpy as np
import random
import cairosvg
from io import BytesIO
from PIL import Image
from pathlib import Path
import albumentations as A
import csv

BASE = Path(__file__).resolve().parent


class AmongUs:
    body_path = (
        "M55.087 40H83c13.807 0 25 11.193 25 25S96.807 90 83 90H52c-.335 0-.668-.007-1-.02V158"
        "a6 6 0 0 0 6 6h9a6 6 0 0 0 6-6v-18a6 6 0 0 1 6-6h24a6 6 0 0 1 6 6v18a6 6 0 0 0 6 6h9a6 6 0 0 0 6-6V54c0-14.36-11.641-26-26-26H77c-9.205 0-17.292 4.783-21.913 12Z"
    )

    holes_paths = [
        "M39 86.358C31.804 81.97 27 74.046 27 65c0-9.746 5.576-18.189 13.712-22.313C45.528 27.225 59.952 16 77 16h26c16.043 0 29.764 9.942 35.338 24H147c9.941 0 18 8.059 18 18v65c0 9.941-8.059 18-18 18h-6v17c0 9.941-8.059 18-18 18h-9c-9.941 0-18-8.059-18-18v-12H84v12c0 9.941-8.059 18-18 18h-9c-9.941 0-18-8.059-18-18V86.358Z",
        "M141 129h6a6 6 0 0 0 6-6V58a6 6 0 0 0-6-6h-6.052c.035.662.052 1.33.052 2v75Z",
        "M52 52c-7.18 0-13 5.82-13 13s5.82 13 13 13h31c7.18 0 13-5.82 13-13s-5.82-13-13-13H52Z",
    ]

    def __init__(self, width=200, height=200, body_color="#ff0000", visor_colors=None):
        self.width = width
        self.height = height
        self.body_color = body_color
        self.visor_colors = visor_colors or ["#ffffff", "#cccccc", "#999999"]

    def to_svg(self):
        svg_body = (
            f'<path d="{self.body_path}" fill="{self.body_color}" fill-rule="evenodd"/>'
        )
        svg_holes = ""

        for i, path in enumerate(self.holes_paths):
            color = self.visor_colors[i] if i < len(self.visor_colors) else "#ffffff"
            svg_holes += f'\n  <path d="{path}" fill="{color}" fill-rule="evenodd"/>'

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 200 200">\n  {svg_body}{svg_holes}\n</svg>'
        )


# ------------------------------------------------------------
# AUGMENTATIONS
# ------------------------------------------------------------

figure_augment = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(
            std_range=[0.1, 0.2],
            mean_range=[0, 0],
            per_channel=True,
            noise_scale_factor=1,
            p=0.5,
        ),
    ],
    p=1.0,
)


def augment_figure_albumentations(image_bgr, image_alpha):
    augmented = figure_augment(image=image_bgr)
    image_bgr_aug = augmented["image"]
    return image_bgr_aug, image_alpha


def random_background(width=800, height=600):
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def svg_to_cv2_image(svg_instance, width, height):
    png_bytes = cairosvg.svg2png(
        bytestring=svg_instance.to_svg().encode("utf-8"),
        output_width=width,
        output_height=height,
    )
    pil_image = Image.open(BytesIO(png_bytes)).convert("RGBA")
    cv_image = np.array(pil_image)

    bgr = cv_image[:, :, :3]
    alpha = cv_image[:, :, 3] / 255.0

    return bgr, alpha


def overlay_image(bg, fg_bgr, fg_alpha, x, y):
    h, w = fg_bgr.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    if x + w > bg_w:
        w = bg_w - x
        fg_bgr = fg_bgr[:, :w]
        fg_alpha = fg_alpha[:, :w]

    if y - h < 0:
        h = y
        fg_bgr = fg_bgr[-h:, :]
        fg_alpha = fg_alpha[-h:, :]

    roi = bg[y - h : y, x : x + w]

    for c in range(3):
        roi[:, :, c] = fg_alpha * fg_bgr[:, :, c] + (1 - fg_alpha) * roi[:, :, c]

    bg[y - h : y, x : x + w] = roi
    return bg


def random_hex_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def load_random_background(folder, width=800, height=600):
    folder = Path(folder)
    paths = list(folder.glob("*"))

    if not paths:
        raise ValueError(f"No background images found in {folder}")

    img_path = random.choice(paths)
    bg = cv2.imread(str(img_path))

    if bg is None:
        raise ValueError(f"Cannot read background image: {img_path}")

    return cv2.resize(bg, (width, height))


# ------------------------------------------------------------
# MAIN GENERATOR WITH BBOX
# ------------------------------------------------------------


def generate(
    destination_folder,
    background_folder=None,
    num_generations=10,
    num_figures=3,
    augment=False,
    random_color=False,
    draw_bbox=False,
    figure_size_range=(80, 200),
):
    """
    Generate images of Among Us characters with random sizes.

    figure_size_range: tuple (min_width, max_width). Height = width * 1.5
    Saves bounding boxes to a CSV file: filename,x_min,y_min,x_max,y_max
    """
    generation_folder = BASE / destination_folder
    (generation_folder / "images").mkdir(parents=True, exist_ok=True)
    csv_file = generation_folder / "images.csv"

    with open(csv_file, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["filename", "x_min", "y_min", "x_max", "y_max"])  # CSV header

        for id in range(num_generations):
            if background_folder:
                bg = load_random_background(BASE / background_folder, 800, 600)
            else:
                bg = random_background(800, 600)

            bg_h, bg_w = bg.shape[:2]
            bboxes = []

            for _ in range(num_figures):
                width = random.randint(figure_size_range[0], figure_size_range[1])
                height = int(width * 1.5)

                body = random_hex_color() if random_color else "#ff0000"
                visors = (
                    [random_hex_color() for _ in range(3)]
                    if random_color
                    else ["#FFFFFF"] * 3
                )
                among_us = AmongUs(
                    width=width, height=height, body_color=body, visor_colors=visors
                )

                fg_bgr, fg_alpha = svg_to_cv2_image(
                    among_us, among_us.width, among_us.height
                )

                if random.random() < 0.5:
                    fg_bgr = cv2.flip(fg_bgr, 1)
                    fg_alpha = cv2.flip(fg_alpha, 1)

                if augment:
                    fg_bgr, fg_alpha = augment_figure_albumentations(fg_bgr, fg_alpha)

                x = random.randint(0, bg_w - width)
                y = random.randint(height, bg_h)

                x_min, y_min = x, y - height
                x_max, y_max = x + width, y
                bboxes.append((x_min, y_min, x_max, y_max))

                bg = overlay_image(bg, fg_bgr, fg_alpha, x, y)

                if draw_bbox:
                    cv2.rectangle(bg, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            img_path = generation_folder / "images" / f"among_us_{id}.png"
            cv2.imwrite(str(img_path), bg)

            for bbox in bboxes:
                writer.writerow([img_path.name, bbox[0], bbox[1], bbox[2], bbox[3]])


if __name__ == "__main__":
    generate(
        "../data/image_train",
        background_folder=None,
        num_generations=100,
        num_figures=5,
        augment=True,
        random_color=True,
        draw_bbox=True,
        figure_size_range=(80, 150),
    )
