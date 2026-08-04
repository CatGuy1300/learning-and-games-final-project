"""GPU and PyTorch device resolution utilities."""

import torch


def get_device(requested_device: str = "auto") -> torch.device:
    """Resolve and return a valid PyTorch device based on user preference and hardware availability.

    Parameters
    ----------
    requested_device : str
        Preferred device string: 'auto', 'cuda', 'cpu', or 'mps'.

    Returns
    -------
    torch.device
        Target PyTorch device object.
    """
    req = requested_device.lower().strip()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available on this system.")
    elif req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available.")
    elif req == "cpu":
        return torch.device("cpu")
    elif req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    else:
        return torch.device(requested_device)


def enable_gpu_optimizations(device: torch.device, fp32_precision: str = "high") -> None:
    """Enable PyTorch CUDA optimizations including TensorFloat-32 (TF32) Tensor Cores on Ampere/Ada GPUs.

    Parameters
    ----------
    device : torch.device
        Target device.
    fp32_precision : str
        Float32 matmul precision ('high' or 'highest').
    """
    if device.type == "cuda":
        torch.set_float32_matmul_precision(fp32_precision)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

def setup_dtype(dtype_str: str = "float32") -> None:
    """Set the global default PyTorch float precision.

    Parameters
    ----------
    dtype_str : str
        Target precision ('float32' or 'float64').
    """
    if dtype_str.lower() == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
