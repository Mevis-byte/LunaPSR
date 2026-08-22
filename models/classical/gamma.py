"""Classical gamma model exposed through the model interface."""

from __future__ import annotations

import numpy as np

from models.ai.base import EnhancementModel, register_model


@register_model
class GammaModel(EnhancementModel):
    name = "gamma"
    category = "classical"
    version = "1.0"
    is_learned = False

    def __init__(self, gamma: float = 0.75, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma = float(gamma)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        from core.enhancement.gamma import gamma_correction

        return gamma_correction(image, gamma=self.gamma)

    def trained_for(self) -> str:
        return "dataset-independent (classical alias)"