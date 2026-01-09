import cv2
import numpy as np
import random
import cairosvg
from io import BytesIO
from PIL import Image
from pathlib import Path
import albumentations as A
import csv
import shutil
from src.data_module.utils import colors
from tqdm.auto import tqdm
from src.configs import DatasetCreationConfig
from cyclopts import App
app=App(name="Generating dataset for training")

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

    def __init__(
        self,
        width=200,
        height=200,
        body_color="#ff0000",
        visor_colors=["#ffffff", "#cccccc", "#999999"],
    ):
        self.width = width
        self.height = height
        self.body_color = body_color
        self.visor_colors = visor_colors

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
        A.CropAndPad(percent=[-0.2, -0.14, -0.23, -0.12], keep_size=False, p=1.0),
        A.HorizontalFlip(p=0.5),
    ],
    p=1.0,
)


def get_crop_and_pad(top, right, bottom, left):
    def crop_mask(mask):
        h, w = mask.shape[:2]
        mask = mask.copy()
        mask[: int(top * h), :] = 0
        mask[h - int(bottom * h) :, :] = 0
        mask[:, : int(w * left)] = 0
        mask[:, w - int(w * right) :] = 0

        return mask

    return crop_mask


def augment_figure_mask(image_bgr):
    augmented = figure_augment(image=image_bgr[..., :3], mask=image_bgr[..., 3])
    image_bgr_aug = np.concatenate(
        (augmented["image"], augmented["mask"][..., None]), axis=-1
    )
    return image_bgr_aug


def augment_mask_only(image_bgr):
    augment_args = [random.uniform(0, 0.3) for _ in range(4)]
    mask = get_crop_and_pad(*augment_args)(mask=image_bgr[..., 3])

    image_bgr_aug = np.concatenate((image_bgr[..., :3], mask[..., None]), axis=-1)

    return image_bgr_aug


def random_background(width=800, height=600):
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def svg_to_cv2_image(svg_instance, width, height):
    png_bytes = cairosvg.svg2png(
        bytestring=svg_instance.to_svg().encode("utf-8"),
        output_width=width,
        output_height=height,
    )
    return np.array(Image.open(BytesIO(png_bytes)).convert("RGBA"))


def overlay_image(bg, fg_img, x, y):
    bgr = fg_img[:, :, :3]
    alpha = fg_img[:, :, 3] / 255.0
    h, w = bgr.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    if x + w > bg_w:
        w = bg_w - x
        bgr = bgr[:, :w]
        alpha = alpha[:, :w]

    if y - h < 0:
        h = y
        bgr = bgr[-h:, :]
        alpha = alpha[-h:, :]

    roi = bg[y - h : y, x : x + w]

    for c in range(3):
        roi[:, :, c] = alpha * bgr[:, :, c] + (1 - alpha) * roi[:, :, c]

    bg[y - h : y, x : x + w] = roi
    return bg


def random_figure_color():
    color_name, color = random.choice(colors)
    color = "#" + "".join(reversed([color[i : i + 2] for i in range(0, 6, 2)]))
    return color_name, color


def random_hex_color():
    return "#{:02x}{:02x}{:02x}".format(
        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    )


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

@app.command
def generate_data(config: DatasetCreationConfig):
    """
    Generate images of Among Us characters with random sizes.

    figure_size_range: tuple (min_width, max_width). Height = width * 1.5
    Saves bounding boxes to a CSV file: filename,x_min,y_min,x_max,y_max
    """
    destination_folder = Path(config.destination_folder)
    if destination_folder.exists():
        shutil.rmtree(destination_folder)
    (destination_folder / "images").mkdir(parents=True, exist_ok=True)
    csv_file = destination_folder / "images.csv"

    with open(csv_file, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            ["filename", "x_min", "y_min", "x_max", "y_max", "figure_color"]
        )  # CSV header
        for id in tqdm(
            range(config.num_generations), desc="Generating...", leave=False
        ):
            if config.background_folder:
                bg = load_random_background(config.background_folder, 800, 600)
            else:
                bg = random_background(800, 600)

            bg_h, bg_w = bg.shape[:2]
            bboxes = []

            for _ in range(random.randint(1, config.num_figures)):
                width = random.randint(
                    config.figure_size_range[0], config.figure_size_range[1]
                )
                height = int(width * 1.5)

                color_name, body = random_figure_color()
                visors = [
                    body,
                    random_hex_color(),
                    random_hex_color(),
                ]  # first is body second is under arm third is eyes
                among_us = AmongUs(
                    width=width, height=height, body_color=body, visor_colors=visors
                )

                fg_img = svg_to_cv2_image(among_us, among_us.width, among_us.height)
                if config.augment_figure:
                    fg_img = augment_figure_mask(fg_img)
                if config.augment_mask:
                    fg_img = augment_mask_only(fg_img)
                height, width = fg_img.shape[:2]
                x = random.randint(0, bg_w - width)
                y = random.randint(height, bg_h)
                x_min, y_min = x, y - height
                x_max, y_max = x + width, y
                bboxes.append((x_min, y_min, x_max, y_max, color_name))
                bg = overlay_image(bg, fg_img, x, y)
                if config.draw_bbox:
                    bbox = bboxes[-1]
                    cv2.rectangle(
                        bg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                    )
            img_path = destination_folder / "images" / f"among_us_{id}.png"
            cv2.imwrite(str(img_path), bg)
            for bbox in bboxes:
                writer.writerow(
                    [img_path.name, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]]
                )
        print("Generating Done")


if __name__ == "__main__":
    app()
    # generate(
    #     DatasetCreationConfig(
    #         "data/image_train",
    #         background_folder=None,
    #         num_generations=100,
    #         num_figures=5,
    #         augment=True,
    #         draw_bbox=True,
    #         figure_size_range=(80, 150),
    #     )
    # )
