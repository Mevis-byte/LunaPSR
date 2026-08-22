"""Classical CLAHE model exposed through the model interface."""

from __future__ import annotations

import numpy as np

from models.ai.base import EnhancementModel, register_model


@register_model
class ClaheModel(EnhancementModel):
    name = "clahe"
    category = "classical"
    version = "1.0"
    is_learned = False

    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clip_limit = float(clip_limit)
        self.tile_size = int(tile_size)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        from core.enhancement.clahe import clahe

        return clahe(image, clip_limit=self.clip_limit, tile_size=self.tile_size)

    def trained_for(self) -> str:
        return "dataset-independent (classical alias)"