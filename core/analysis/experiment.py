"""Experiment recording and derived-output storage (Phase 3).

Every enhancement result can be recorded as a reproducible JSON document:

    {
      "product": "ch2_..._d_img_d18",
      "analysis": {"factor": 10, "array_pixels": ...},
      "method": "clahe",
      "parameters": {...},
      "normalization": {"lower_percentile": 1, "upper_percentile": 99,
                        "lower_value": ..., "upper_value": ...},
      "metrics": {"original": {...}, "enhanced": {...}},
      "source": "...",
      "timestamp": "..."
    }

DERIVED OUTPUTS GO TO ``outputs/`` -- never into the OHRC product directory
(which stays read-only).  Callers may also save derived 8-bit preview PNGs and
metrics JSON beside the record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from configs.defaults import EXPERIMENTS_DIR

__all__ = ["ExperimentRecord", "ExperimentStore"]

_DT_FORMAT = "%Y%m%dT%H%M%SZ"


def _as_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict") and not isinstance(obj, (int, float, str, bool)):
        return _as_jsonable(obj.to_dict())
    return obj


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ExperimentRecord:
    """One reproducible lab experiment."""

    product: str
    method: str
    parameters: dict = field(default_factory=dict)
    normalization: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    noise_model: dict = field(default_factory=dict)
    source: str = "LunaPSR/lab"
    timestamp: str = field(default_factory=_utc_now)
    notes: Optional[str] = None

    def identifier(self) -> str:
        """A short, deterministic id used for filenames."""
        stem = ".".join(
            [x.lower() for x in (self.product, self.method)]
            + [f"p{int(self.normalization.get('lower_percentile', 1))}-"
               f"{int(self.normalization.get('upper_percentile', 99))}"]
        )
        clean = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
        return f"{clean.replace('.', '_')}_{self.timestamp}"

    def to_dict(self) -> dict:
        return {
            "product": self.product,
            "method": self.method,
            "parameters": _as_jsonable(self.parameters),
            "normalization": _as_jsonable(self.normalization),
            "analysis": _as_jsonable(self.analysis),
            "metrics": _as_jsonable(self.metrics),
            "noise_model": _as_jsonable(self.noise_model),
            "source": self.source,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class ExperimentStore:
    """Persists experiment records and derived previews under ``outputs/``."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else EXPERIMENTS_DIR.parent
        self.experiments = self.root / "experiments"
        self.previews = self.root / "previews"
        self.metrics = self.root / "metrics"
        for d in (self.experiments, self.previews, self.metrics):
            d.mkdir(parents=True, exist_ok=True)

    def save_record(self, record: ExperimentRecord) -> Path:
        out = self.experiments / f"{record.identifier()}.json"
        out.write_text(record.to_json(), encoding="utf-8")
        return out

    def save_preview(self, name: str, arr8: Any) -> Path:
        """Save a derived 8-bit visualisation PNG (labelled derived)."""
        import numpy as np
        from PIL import Image

        out = self.previews / f"{name}.png"
        Image.fromarray(np.asarray(arr8, dtype=np.uint8)).save(out, format="PNG")
        return out

    def save_metrics(self, name: str, metrics: dict) -> Path:
        out = self.metrics / f"{name}.json"
        out.write_text(json.dumps(_as_jsonable(metrics), indent=2, default=str),
                       encoding="utf-8")
        return out

    def list_records(self) -> list:
        return sorted(p.name for p in self.experiments.glob("*.json"))


def save_experiment(record: ExperimentRecord, store: Optional[ExperimentStore] = None) -> dict:
    """Persist a record and return its locations + id."""
    store = store or ExperimentStore()
    path = store.save_record(record)
    return {"id": record.identifier(), "path": str(path)}