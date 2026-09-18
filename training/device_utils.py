"""Picks the fastest available torch device. Capability checks
(`is_available()`) rather than `platform.system()` branching -- that's the
robust way to do this (an Intel Mac has no MPS, a Linux box might have no
CUDA either), and it's what `torch.backends.mps.is_available()` /
`torch.cuda.is_available()` already do under the hood."""

import torch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
