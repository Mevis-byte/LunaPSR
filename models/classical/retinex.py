"""Classical Single-Scale Retinex model."""

from __future__ import annotations

import numpy as np

from models.ai.base import EnhancementModel, register_model


@register_model
class RetinexModel(EnhancementModel):
    name = "retinex"
    category = "classical"
    version = "1.0"
    is_learned = False

    def __init__(self, sigma: float = 30.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sigma = float(sigma)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        from core.enhancement.retinex import single_scale_retinex

        return single_scale_retinex(image, sigma=self.sigma)

    def trained_for(self) -> str:
        return "dataset-independent (classical alias)"