import torch
from typing import Tuple

def fcos_loss_fn(predictions:dict[str, torch.Tensor], targets:dict[str, torch.Tensor], lambdas:Tuple[float, float, float]):
    """
    FCOS model returns a dictionary with losses when in training mode.
    This function extracts and sums the total loss.
    """
    total_loss = lambdas[0]*predictions['classification']+lambdas[1]*predictions['bbox_regression']+lambdas[2]*predictions['bbox_ctrness']
    return total_loss
