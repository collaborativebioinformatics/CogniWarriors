"""Model-weight serialization used at the federation boundary."""

from __future__ import annotations

import base64
import io
from typing import Dict, Optional

import torch


WEIGHTS_FORMAT = "torch_state_dict_base64"


def encode_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    cpu_state = {key: value.detach().cpu() for key, value in state_dict.items()}
    torch.save(cpu_state, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_state_dict(payload: Optional[str]) -> Optional[Dict[str, torch.Tensor]]:
    if not payload:
        return None

    try:
        raw = base64.b64decode(payload.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("weights must be base64-encoded") from exc

    buffer = io.BytesIO(raw)
    try:
        try:
            loaded = torch.load(buffer, map_location="cpu", weights_only=True)
        except TypeError:
            buffer.seek(0)
            loaded = torch.load(buffer, map_location="cpu")
    except Exception as exc:
        raise ValueError("weights could not be decoded as a torch state_dict") from exc

    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise ValueError("weights payload must contain a state_dict")
    return loaded

