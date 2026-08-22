"""Processing history.

Every processed image yields a :class:`ProcessingRecord` that can be exported
as JSON and re-run to produce the same result.  Records are persisted to
``data/history`` so the session and past sessions are reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from configs.defaults import HISTORY_DIR

__all__ = [
    "ProcessingRecord",
    "ProcessingHistory",
    "new_processing_id",
    "utc_now_iso",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_processing_id(year: int | None = None, existing: Optional[set] = None) -> str:
    """Generate a sequential ID like ``PSR-2026-00001``."""
    year = year or datetime.now(timezone.utc).year
    existing = existing or set()
    prefix = f"PSR-{year}"
    counter = 1
    while f"{prefix}-{counter:05d}" in existing:
        counter += 1
    return f"{prefix}-{counter:05d}"


@dataclass
class ProcessingRecord:
    """Full reproducible record of one processing run."""

    processing_id: str
    timestamp: str = field(default_factory=utc_now_iso)
    input_name: str = ""
    input_summary: dict = field(default_factory=dict)
    pipeline_name: str = "manual"
    pipeline_stages: list = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "processing_id": self.processing_id,
            "timestamp": self.timestamp,
            "input": {
                "source_name": self.input_name,
                **self.input_summary,
            },
            "pipeline": {
                "name": self.pipeline_name,
                "stages": self.pipeline_stages,
            },
            "parameters": self._flatten_parameters(),
            "warnings": self.warnings,
            "metrics": self.metrics,
            "notes": self.notes,
            "reproducibility_notes": self._reproducibility_notes(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, directory: Optional[Path] = None) -> Path:
        directory = Path(directory) if directory else HISTORY_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.processing_id}.json"
        path.write_text(self.to_json())
        return path

    def _flatten_parameters(self) -> dict:
        params: dict = {}
        for stage in self.pipeline_stages:
            label = stage.get("stage", "?")
            params[label] = stage.get("parameters", {})
        return params

    def _reproducibility_notes(self) -> str:
        return (
            "Re-run: load the same source image and apply the pipeline stages "
            "with the recorded parameters (plus the recorded normalization) "
            "to reproduce this result.  Display normalization does not alter "
            "the underlying numerics."
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingRecord":
        inp = dict(data.get("input", {}))
        inp_name = inp.pop("source_name", "")

        # JSON keys cannot collide: source_name was read out already.
        rec = cls(
            processing_id=data["processing_id"],
            timestamp=data.get("timestamp", ""),
            input_name=inp_name,
            input_summary=inp,
            pipeline_name=data.get("pipeline", {}).get("name", "manual"),
            pipeline_stages=data.get("pipeline", {}).get("stages", []),
            warnings=data.get("warnings", []),
            metrics=data.get("metrics", {}),
            notes=data.get("notes", ""),
        )
        return rec


class ProcessingHistory:
    """In-memory + on-disk history of processing runs."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = Path(directory) if directory else HISTORY_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records: List[ProcessingRecord] = []
        self._load()

    def _load(self) -> None:
        for path in sorted(self.directory.glob("PSR-*.json")):
            try:
                data = json.loads(path.read_text())
                self.records.append(ProcessingRecord.from_dict(data))
            except Exception:
                continue
        self.records.sort(key=lambda r: r.timestamp, reverse=True)

    @property
    def used_ids(self) -> set:
        return {r.processing_id for r in self.records}

    def next_id(self) -> str:
        return new_processing_id(existing=self.used_ids)

    def add(self, record: ProcessingRecord) -> ProcessingRecord:
        """Add (deduplicated by id) and persist to disk."""
        self.records = [r for r in self.records if r.processing_id != record.processing_id]
        self.records.insert(0, record)
        record.save(self.directory)
        return record

    def get(self, processing_id: str) -> Optional[ProcessingRecord]:
        for r in self.records:
            if r.processing_id == processing_id:
                return r
        return None