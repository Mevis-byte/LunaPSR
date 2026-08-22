"""Retinexformer model stub.

Retinexformer (Cai et al.) is a low-light enhancement transformer for *generic
photography*; even when integrated it must be validated for OHRC PSR imagery
before any scientific use.  We keep the loading path lazy so enabling it later
does not require code changes.
"""

from __future__ import annotations

import numpy as np

from models.ai.base import EnhancementModel


class RetinexformerModel(EnhancementModel):
    name = "retinexformer"
    category = "ai"
    version = "0.0.1"
    is_learned = True
    _weights_url = ""  # not bundled in Version 1

    def load(self) -> "RetinexformerModel":
        raise NotImplementedError(
            "Retinexformer weights are not bundled in LunaPSR Version 1. "
            "Model loading will be enabled in Phase 4 of the research roadmap."
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Retinexformer is not available yet. See docs/research-roadmap.md Phase 4."
        )

    def trained_for(self) -> str:
        return "Generic low-light photography (e.g. LOL); NOT validated for OHRC/PSR imagery"

    def available(self) -> bool:
        return False