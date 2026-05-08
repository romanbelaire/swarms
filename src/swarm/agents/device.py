"""Global PyTorch device for swarm agents (default: CUDA when available)."""

import os

import torch


def resolve_training_device() -> torch.device:
    """
    Pick device for networks and batched tensors.

    Override with env ``SWARM_DEVICE`` (e.g. ``cpu``, ``cuda``, ``cuda:1``, ``mps``).
    Otherwise: CUDA if available, else Apple MPS if available, else CPU.
    """
    override = os.environ.get("SWARM_DEVICE", "").strip()
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = resolve_training_device()

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
