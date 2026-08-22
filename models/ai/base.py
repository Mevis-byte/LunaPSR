"""AI model interfaces.

Defines the common :class:`EnhancementModel` contract so classical baselines
and future learned models (Zero-DCE, Zero-DCE++, Retinexformer, physics-aware
PSR models) can be benchmarked side by side with identical semantics.

Models are **lazy**: loading a model (weights download, GPU setup) happens only
when the user actually selects it, via :meth:`EnhancementModel.load`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

__all__ = ["EnhancementModel", "MODEL_REGISTRY"]

ModelName = str


class EnhancementModel(ABC):
    """Common interface for any enhancement algorithm (classical or AI).

    ``enhance`` receives a **normalised [0,1] float32 array** and returns one.
    Providers MUST NOT claim calibrated scientific validity unless their
    documentation explicitly supports it.
    """

    name: ModelName = "base"
    category: str = "base"
    version: str = "0.0.0"
    is_learned: bool = False

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or f"{self.category}:{self.name}"
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> "EnhancementModel":
        """Load weights / initialise the model. No-op for pure classical."""
        self._loaded = True
        return self

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    # -- interface ---------------------------------------------------------
    @abstractmethod
    def enhance(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def metadata(self) -> Dict:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "category": self.category,
            "version": self.version,
            "is_learned": self.is_learned,
            "trained_for": self.trained_for(),
        }

    def trained_for(self) -> str:
        """Dataset/instrument the model is trained/validated for, or 'n/a'."""
        return "n/a"

    def available(self) -> bool:
        """Whether the model can be used in this install (deps present)."""
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.model_id} loaded={self._loaded}>"


MODEL_REGISTRY: Dict[ModelName, type] = {}


def register_model(cls: type) -> type:
    """Register a model class under its `name`."""
    instance_probe = cls.__new__(cls)
    key = getattr(instance_probe, "name", cls.__name__)
    MODEL_REGISTRY[key] = cls
    return cls


def get_model(model_id: ModelName, **kwargs) -> EnhancementModel:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_id}'. Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_id](**kwargs)