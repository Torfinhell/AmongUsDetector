import torch


def fcos_loss_fn(predictions, targets):
    """
    FCOS model returns a dictionary with losses when in training mode.
    This function extracts and sums the total loss.
    """
    if isinstance(predictions, dict) and "loss_classifier" in predictions:
        # Model is in training mode and returns losses
        total_loss = sum(
            loss for loss in predictions.values() if isinstance(loss, torch.Tensor)
        )
        return total_loss
    else:
        # Fallback: use smooth L1 loss if needed
        raise RuntimeError("FCOS model should return loss dict in training mode")
