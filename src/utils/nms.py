import torch
from .iou import iou_score


def nms(preds, iou_thr=0.9):
    pred_output = []
    keys = ["boxes", "scores", "labels"]

    for pred_image in preds:
        pred = list(zip(*pred_image.values()))
        pred = sorted(pred, key=lambda x: -x[1])

        nms_preds = []
        for bbox_pred in pred:
            if not any(
                iou_score(nms_pred[0], bbox_pred[0]) > iou_thr for nms_pred in nms_preds
            ):
                nms_preds.append(bbox_pred)

        if nms_preds:
            pred_output.append(
                {key: torch.stack(val) for key, val in zip(keys, zip(*nms_preds))}
            )
        else:
            pred_output.append({key: torch.empty(0) for key in keys})

    return pred_output
