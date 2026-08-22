"""PSR-aware processing.

Level 1 (this version): a :class:`PSRDetector` interface plus a manual ROI
implementation.  Future physics-aware detection will use illumination
geometry (solar incidence, lunar coordinates, acquisition time, DEM/shadow
casting) -- we absolutely do **not** fabricate PSR detections from pixel stats
alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PSRRegion:
    """A detected / user-provided Permanently Shadowed Region ROI."""

    x0: int
    y0: int
    x1: int
    y1: int
    label: str = ""
    source: str = "manual"
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return max(self.x1 - self.x0, 0)

    @property
    def height(self) -> int:
        return max(self.y1 - self.y0, 0)

    def as_slice(self) -> tuple:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


class PSRDetector(ABC):
    """Interface for PSR detection.

    Future implementations will combine optical imagery with:
    * solar incidence / illumination geometry
    * lunar latitude / longitude and DEM
    * acquisition time and sensor geometry
    * crater-shape based shadow casting

    Until then, only :class:`ManualPSRDetector` exists.
    """

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name

    @abstractmethod
    def detect(self, image: np.ndarray, metadata: Optional[dict] = None) -> List[PSRRegion]:
        """Return a list of PSR regions detected in ``image``."""
        raise NotImplementedError


class ManualPSRDetector(PSRDetector):
    """Detector that returns a user-supplied ROI.

    Used to prototype ROI-based analysis and the PSR enhancement workflow
    before physics-aware detection is implemented.
    """

    def __init__(self, region: Optional[PSRRegion] = None) -> None:
        super().__init__(name="manual")
        self.region = region

    def detect(self, image: np.ndarray, metadata: Optional[dict] = None) -> List[PSRRegion]:
        if self.region is None:
            return []
        reg = self.region
        h, w = image.shape[:2]
        # clamp to image bounds
        x0, y0 = min(max(reg.x0, 0), w), min(max(reg.y0, 0), h)
        x1, y1 = min(max(reg.x1, 0), w), min(max(reg.y1, 0), h)
        return [
            PSRRegion(
                x0=x0, y0=y0, x1=max(x1, x0 + 1), y1=max(y1, y0 + 1),
                label=reg.label or "PSR ROI",
                source="manual",
                confidence=0.0,
                metadata={"note": "Manually selected ROI; not a physics-based PSR detection."},
            )
        ]