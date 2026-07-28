

import torch
import pydantic

def get_device() -> torch.device:
    """Return the device to use for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """Return the dtype to use for inference."""
    if device.type in ["cuda", "mps"]:
        return torch.float16
    else:
        return torch.float32

