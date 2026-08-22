"""Zero-DCE / Zero-DCE++ model stub.

Zero-DCE++ (Guo et al.) is a lightweight, reference-free low-light enhancer.
Like Retinexformer it is trained on generic photography and must be
re-validated for OHRC PSR data before scientific claims.  Lazy loading only.
"""

from __future__ import annotations

import numpy as np

from models.ai.base import EnhancementModel


class ZeroDCEModel(EnhancementModel):
    name = "zero_dce"
    category = "ai"
    version = "0.0.1"
    is_learned = True

    def load(self) -> "ZeroDCEModel":
        raise NotImplementedError(
            "Zero-DCE weights are not bundled in LunaPSR Version 1. "
            "Model loading will be enabled in Phase 4 of the research roadmap."
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Zero-DCE is not available yet. See docs/research-roadmap.md Phase 4."
        )

    def trained_for(self) -> str:
        return "Generic low-light photography; NOT validated for OHRC/PSR imagery"

    def available(self) -> bool:
        return False