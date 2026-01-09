from torchmetrics import Metric
import torch
from src.configs import MetricConfig
from src.utils import iou_score


class PRauc(Metric):
    def __init__(self, config: MetricConfig):
        super().__init__()
        self.add_state("all_scores", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("all_taken", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("N_gt", default=torch.tensor(0), dist_reduce_fx="sum")
        self.iou_thr = config.mAP_iou_thr

    def update(self, preds, targets):
        assert len(preds) == len(targets)
        all_scores = []
        all_taken = []
        N_gt = 0
        for pred, target in zip(preds, targets):
            gt = list(
                zip(
                    target["boxes"].detach().tolist(),
                    target["labels"].detach().tolist(),
                )
            )
            N_gt += len(target["boxes"])
            pred = list(zip(pred["boxes"], pred["scores"], pred["labels"]))
            pred = sorted(pred, key=lambda x: -x[1])
            for detection in pred:
                gt_label = [
                    (ind, item[0])
                    for ind, item in enumerate(gt)
                    if detection[2] == item[1]
                ]
                if not len(gt_label):
                    all_scores.append(detection[1])
                    all_taken.append(0)
                    continue
                gt_label_ind, gt_label_bboxes = zip(*gt_label)
                calc_ious = [
                    iou_score(detection[0], gt_bbox) for gt_bbox in gt_label_bboxes
                ]
                if max(calc_ious) >= self.iou_thr:
                    all_scores.append(detection[1])
                    all_taken.append(1)
                    del gt[gt_label_ind[calc_ious.index(max(calc_ious))]]
                else:
                    all_scores.append(detection[1])
                    all_taken.append(0)
        if all_scores:
            self.all_scores = torch.cat(
                [
                    self.all_scores,
                    torch.tensor(all_scores, device=self.all_scores.device),
                ]
            )
            self.all_taken = torch.cat(
                [
                    self.all_taken,
                    torch.tensor(all_taken, device=self.all_scores.device),
                ]
            )
        self.N_gt += N_gt

    def compute(self):
        if self.N_gt == 0 or self.all_scores.numel() == 0:
            return torch.tensor(0.0, device=self.all_scores.device)
        sorted_idx = torch.argsort(self.all_scores, descending=True)
        taken = self.all_taken[sorted_idx]
        tp = torch.cumsum(taken, dim=0)
        fp = torch.cumsum(1 - taken, dim=0)
        recall = tp / self.N_gt
        precision = tp / (tp + fp)
        recall = torch.cat([torch.tensor([0.0], device=recall.device), recall])
        precision = torch.cat([torch.tensor([1.0], device=precision.device), precision])
        auc = torch.trapz(precision, recall)
        return auc
