"""Random seed reproducibility manager."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set global random seed for Python, NumPy, and PyTorch (CPU and CUDA).

    Parameters
    ----------
    seed : int
        Seed value to initialize random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
