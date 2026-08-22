"""Enhancement pipeline engine.

A pipeline is an ordered list of stages.  Each stage runs a registered
function and records reproducible metadata (stage name + parameters).  The
result carries the processed [0,1] array, the per-stage metadata, and any
warnings.

Example
-------
>>> pipeline = EnhancementPipeline()
>>> pipeline.add("normalize", percentiles=(0.5, 99.5))
>>> pipeline.add("gamma", gamma=0.75)
>>> pipeline.add("clahe", clip_limit=2.0, tile_size=8)
>>> result = pipeline.run(image_dn_float32)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.registry import STAGES, validate_stage

__all__ = ["EnhancementPipeline", "PipelineResult", "PipelineStage"]


@dataclass
class PipelineStage:
    """A single recorded stage in a pipeline."""

    name: str
    parameters: dict
    warning: Optional[str] = None

    def metadata(self) -> dict:
        return {"stage": self.name, "parameters": dict(self.parameters)}


@dataclass
class PipelineResult:
    """Output of :meth:`EnhancementPipeline.run`."""

    image: np.ndarray            # [0,1] float32
    stages: List[PipelineStage] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    normalization: dict = field(default_factory=dict)

    def stage_metadata_list(self) -> list:
        return [s.metadata() for s in self.stages]

    def to_dict(self) -> dict:
        return {
            "image_range": [float(np.min(self.image)), float(np.max(self.image))],
            "stages": self.stage_metadata_list(),
            "warnings": self.warnings,
            "normalization": dict(self.normalization),
        }


class EnhancementPipeline:
    """Ordered, reproducible image-processing pipeline."""

    def __init__(self, name: str = "manual") -> None:
        self.name = name
        self.stages: List[PipelineStage] = []

    def add(self, stage_name: str, warning: Optional[str] = None, **params) -> "EnhancementPipeline":
        validate_stage(stage_name)
        defaults = dict(STAGES[stage_name]["defaults"])
        merged = {**defaults, **{k: v for k, v in params.items() if v is not None}}
        self.stages.append(PipelineStage(name=stage_name, parameters=merged, warning=warning))
        return self

    def remove(self, index: int) -> None:
        del self.stages[index]

    @property
    def stage_names(self) -> List[str]:
        return [s.name for s in self.stages]

    def stages_metadata(self) -> list:
        """Serialisable metadata for all stages (name + parameters)."""
        return [s.metadata() for s in self.stages]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stages": self.stages_metadata(),
        }

    def from_dict(self, data: dict) -> "EnhancementPipeline":
        self.name = data.get("name", "manual")
        self.stages = []
        for md in data.get("stages", []):
            self.stages.append(PipelineStage(name=md["stage"], parameters=dict(md.get("parameters", {}))))
        return self

    def run(self, image: np.ndarray) -> PipelineResult:
        """Run the pipeline.

        ``image`` should be a raw float32 (DN-scale) array.  Including a
        ``normalize`` stage maps it into [0,1] before enhancement; if the
        caller already supplies [0,1] data, omit it.
        """
        img = np.asarray(image, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError(f"Pipeline expects a 2-D image, got shape {img.shape}")
        if not np.isfinite(img).all():
            raise ValueError("Pipeline input contains NaN/Inf values")

        result = PipelineResult(image=img)
        for stage in self.stages:
            stage_fn = STAGES[stage.name]["run"]
            stage_params = copy.deepcopy(stage.parameters)
            try:
                img = stage_fn(img, **stage_params)
            except TypeError:
                img = stage_fn(img, **{k: v for k, v in stage_params.items() if k not in ("percentiles", "method")})
            img = np.clip(img, 0.0, 1.0).astype(np.float32)
            if not np.isfinite(img).all():
                raise RuntimeError(
                    f"Stage '{stage.name}' produced NaN/Inf values; refusing to continue."
                )
            result.stages.append(stage)
            if stage.warning:
                result.warnings.append(stage.warning)

        result.image = img
        for stage in self.stages:
            if stage.name == "normalize":
                result.normalization = {"stage": "normalize", **dict(stage.parameters)}
                break
        return result