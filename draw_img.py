import cv2
import numpy as np
import random
import cairosvg
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------

# AmongUs class (from previous code)

# ---------------------------

class AmongUs:
body_path = ("M55.087 40H83c13.807 0 25 11.193 25 25S96.807 90 83 90H52c-.335 0-.668-.007-1-.02V158"
"a6 6 0 0 0 6 6h9a6 6 0 0 0 6-6v-18a6 6 0 0 1 6-6h24a6 6 0 0 1 6 6v18a6 6 0 0 0 6 6h9a6 6 0 0 0 6-6V54c0-14.36-11.641-26-26-26H77c-9.205 0-17.292 4.783-21.913 12Z")

```
holes_paths = [
    "M39 86.358C31.804 81.97 27 74.046 27 65c0-9.746 5.576-18.189 13.712-22.313C45.528 27.225 59.952 16 77 16h26c16.043 0 29.764 9.942 35.338 24H147c9.941 0 18 8.059 18 18v65c0 9.941-8.059 18-18 18h-6v17c0 9.941-8.059 18-18 18h-9c-9.941 0-18-8.059-18-18v-12H84v12c0 9.941-8.059 18-18 18h-9c-9.941 0-18-8.059-18-18V86.358Z",
    "M141 129h6a6 6 0 0 0 6-6V58a6 6 0 0 0-6-6h-6.052c.035.662.052 1.33.052 2v75Z",
    "M52 52c-7.18 0-13 5.82-13 13s5.82 13 13 13h31c7.18 0 13-5.82 13-13s-5.82-13-13-13H52Z"
]

def __init__(self, width=100, height=150, body_color="#ff0000", visor_colors=None):
    self.width = width
    self.height = height
    self.body_color = body_color
    self.visor_colors = visor_colors or ["#ffffff", "#cccccc", "#999999"]

def to_svg(self):
    svg_body = f'<path d="{self.body_path}" fill="{self.body_color}" fill-rule="evenodd"/>'
    svg_holes = ""
    for i, path in enumerate(self.holes_paths):
        color = self.visor_colors[i] if i < len(self.visor_colors) else "#ffffff"
        svg_holes += f'\n  <path d="{path}" fill="{color}" fill-rule="evenodd"/>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 200 200">\n  {svg_body}{svg_holes}\n</svg>')
```

# ---------------------------

# Convert SVG to CV2 image with alpha

# ---------------------------

def svg_to_cv2_image(svg_instance, width, height):
png_bytes = cairosvg.svg2png(bytestring=svg_instance.to_svg().encode('utf-8'),
output_width=width, output_height=height)
pil_image = Image.open(BytesIO(png_bytes)).convert("RGBA")
cv_image = np.array(pil_image)
bgr = cv_image[:, :, :3]
alpha = cv_image[:, :, 3] / 255.0
return bgr, alpha

# ---------------------------

# Overlay FG image with alpha onto BG at bottom-left (x, y)

# Returns bounding box as well

# ---------------------------

def overlay_image_with_bbox(bg, fg_bgr, fg_alpha, x, y):
h, w = fg_bgr.shape[:2]
bg_h, bg_w = bg.shape[:2]

```
# Clip to image boundaries
if x + w > bg_w:
    w = bg_w - x
    fg_bgr = fg_bgr[:, :w]
    fg_alpha = fg_alpha[:, :w]
if y - h < 0:
    h = y
    fg_bgr = fg_bgr[-h:, :]
    fg_alpha = fg_alpha[-h:, :]

# Overlay
for c in range(3):
    bg[y-h:y, x:x+w, c] = fg_alpha * fg_bgr[:, :, c] + (1 - fg_alpha) * bg[y-h:y, x:x+w, c]

# Return bounding box: (x_min, y_min, x_max, y_max)
bbox = (x, y-h, x+w, y)
return bg, bbox
```

# ---------------------------

# Random background

# ---------------------------

def random_background(width=800, height=600):
return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

# ---------------------------

# Add multiple random Among Us characters

# ---------------------------

def add_random_among_us(bg, n_characters=5):
bboxes = []
for _ in range(n_characters):
width = random.randint(50, 120)
height = int(width * 1.5)
body_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
visor_colors = ["#ffffff", "#cccccc", "#999999"]

```
    character = AmongUs(width=width, height=height, body_color=body_color, visor_colors=visor_colors)
    fg_bgr, fg_alpha = svg_to_cv2_image(character, character.width, character.height)

    x = random.randint(0, bg.shape[1] - character.width)
    y = random.randint(character.height, bg.shape[0])

    bg, bbox = overlay_image_with_bbox(bg, fg_bgr, fg_alpha, x, y)
    bboxes.append(bbox)
return bg, bboxes
```

# ---------------------------

# Generate scene

# ---------------------------

bg = random_background(800, 600)
final_img, bboxes = add_random_among_us(bg, n_characters=5)

# ---------------------------

# Display using matplotlib

# ---------------------------

final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.imshow(final_img_rgb)
plt.axis('off')

# Draw bounding boxes

for bbox in bboxes:
x_min, y_min, x_max, y_max = bbox
rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
edgecolor='yellow', facecolor='none', linewidth=2)
plt.gca().add_patch(rect)

plt.show()

# ---------------------------

# Bboxes are now in `bboxes` list for multiple object tracking

# ---------------------------

print("Bounding boxes:", bboxes)
