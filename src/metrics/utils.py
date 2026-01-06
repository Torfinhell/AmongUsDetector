import torch


def iou_score(first_bbox: list[float], second_bbox: list[float]):
    assert (
        len(first_bbox) == 4 and len(second_bbox) == 4
    ), "length of bbox coordinates should be 4"

    x1, y1, x2, y2 = first_bbox
    x1_, y1_, x2_, y2_ = second_bbox
    inter_w = max(0, min(x2, x2_) - max(x1, x1_))
    inter_h = max(0, min(y2, y2_) - max(y1, y1_))
    intersect = inter_w * inter_h
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_ - x1_) * (y2_ - y1_)
    union = area1 + area2 - intersect
    return intersect / union if union > 0 else 0.0


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
