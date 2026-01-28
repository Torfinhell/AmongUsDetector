import torch
import cv2
import numpy as np
import PIL.Image
from pathlib import Path
from src.data_module.utils import color_to_ind, colors, hex_to_bgr
def drawbboxes(images:list[torch.tensor], preds:list[torch.tensor]):
    output_images=[]
    for i, pred in enumerate(preds):
        img=images[i].detach().cpu().numpy()
        img=np.transpose(img, (1, 2, 0))
        img=(img*255).astype(np.uint8)
        for label, pred_bbox in zip(pred["labels"], pred["boxes"]):
            xmin, ymin, xmax, ymax = pred_bbox.detach().cpu().int().tolist()
            hex_color=colors[label][1]
            cv2.rectangle(img, (xmin, ymin),(xmax, ymax),
                color=hex_to_bgr(hex_color))
        output_images.append(torch.from_numpy(img).permute(2, 0, 1))
    return output_images
def create_output(images_paths, preds, resize_shape, output_folder):
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    for batch in zip(images_paths, preds):
        for image_path, pred in zip(batch[0], batch[1]):
            image = np.array(PIL.Image.open(image_path).convert("RGB"))
            for label, (xmin, ymin, xmax, ymax)  in zip(pred["labels"],pred["boxes"]):
                xmin=int(xmin*image.shape[0]/resize_shape[0])
                ymin=int(ymin*image.shape[1]/resize_shape[1])
                xmax=int(xmax*image.shape[0]/resize_shape[0])
                ymax=int(ymax*image.shape[1]/resize_shape[1])
                hex_color=colors[label][1]
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                          color=hex_to_bgr(hex_color),thickness=2)
            cv2.imwrite(Path(output_folder)/Path(image_path).name, image)
            
            