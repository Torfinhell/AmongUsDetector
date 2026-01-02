import torch
import random
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    This sets seeds for Python, NumPy and PyTorch (CPU and CUDA where available).
    It also configures a reasonable matmul precision for float32 operations.
    """
    torch.set_float32_matmul_precision("medium")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
