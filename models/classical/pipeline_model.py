"""Classical pipeline model: any staged enhancement as a model.

Lets the pipeline engine be benchmarked through the same interface as future
AI models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.pipeline import EnhancementPipeline
from models.ai.base import EnhancementModel, register_model


@register_model
class PipelineModel(EnhancementModel):
    name = "pipeline"
    category = "classical"
    version = "1.0"
    is_learned = False

    def __init__(self, pipeline: Optional[EnhancementPipeline] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pipeline = pipeline or EnhancementPipeline(name="model")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        result = self.pipeline.run(image)
        return result.image

    def metadata(self) -> dict:
        md = super().metadata()
        md["pipeline"] = self.pipeline.to_dict()
        return md

    def trained_for(self) -> str:
        return "dataset-independent (staged classical pipeline)"