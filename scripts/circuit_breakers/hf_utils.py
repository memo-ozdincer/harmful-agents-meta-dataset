"""Hugging Face Hub helper utilities.

Centralizes auth token resolution for gated models (e.g., Llama family)
so we don't rely on hardcoded secrets or per-call inconsistencies.

Never commit tokens to the repo. Use env vars instead.
"""

from __future__ import annotations

import os
from typing import Optional


def resolve_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Resolve a Hugging Face access token.

    Precedence:
    1) explicit_token (caller-provided)
    2) HF_TOKEN
    3) HUGGINGFACE_HUB_TOKEN
    4) HUGGINGFACE_TOKEN (legacy)

    Returns None if no token is available.
    """
    if explicit_token:
        return explicit_token

    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
