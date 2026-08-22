from pathlib import Path

import numpy as np
import pytest

import app.exporters as exporters
from app.exporters import (difference_png_bytes, json_bytes, map_png_bytes,
                           png_bytes, tiff16_bytes)
from core.history import ProcessingHistory, ProcessingRecord, new_processing_id
from core.io import (save_16bit_tiff, save_8bit_png, save_numpy)
from core.preprocessing.normalization import robust_normalize
from PIL import Image


def test_png_bytes():
    img = np.linspace(0, 1, 128 * 64).reshape(128, 64).astype(np.float32)
    data = png_bytes(img)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_tiff16_bytes_preserves_range():
    img = np.linspace(0, 1, 128 * 64).reshape(128, 64).astype(np.float32)
    data = tiff16_bytes(img, {"processing_id": "PSR-2026-00001"})
    from io import BytesIO

    arr = np.asarray(Image.open(BytesIO(data)))
    assert arr.dtype == np.uint16
    assert arr.max() > 64000
    assert arr.min() == 0


def test_difference_png_bytes():
    a = np.zeros((50, 50), dtype=np.float32)
    b = np.ones((50, 50), dtype=np.float32)
    data = difference_png_bytes(a, b)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_map_png_bytes_handles_nan():
    m = np.full((40, 40), 0.5, dtype=np.float32)
    m[0, 0] = np.nan
    data = map_png_bytes(m)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_json_bytes():
    raw = json_bytes({"a": 1, "b": [1, 2]})
    import json

    assert json.loads(raw) == {"a": 1, "b": [1, 2]}


def test_save_8bit_png(tmp_path):
    out = save_8bit_png(np.linspace(0, 255, 100).reshape(10, 10), tmp_path / "v.png")
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_16bit_tiff(tmp_path):
    arr = np.linspace(0, 65535, 100).reshape(10, 10)
    out = save_16bit_tiff(arr, tmp_path / "x.tiff", {"processing_id": "PSR-1"})
    assert out.exists()


def test_save_numpy_roundtrip(tmp_path):
    arr = np.random.rand(20, 20).astype(np.float32)
    out = save_numpy(arr, tmp_path / "x.npz")
    loaded = np.load(out)["data"]
    assert np.allclose(loaded, arr)


def test_tiff_rejects_nan():
    bad = np.full((10, 10), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        save_16bit_tiff(bad, "/tmp/nope.tiff")
    with pytest.raises(ValueError):
        tiff16_bytes(bad)


def test_history_save_load(tmp_path):
    rec = ProcessingRecord(
        processing_id="PSR-2026-00099",
        input_name="ch2_ohr.png",
        pipeline_name="gamma_clahe",
        pipeline_stages=[{"stage": "gamma", "parameters": {"gamma": 0.75}}],
        warnings=["proxy note"],
        metrics={"original": {}, "enhanced": {}},
    )
    rec.save(tmp_path)
    history = ProcessingHistory(directory=tmp_path)
    loaded = history.get("PSR-2026-00099")
    assert loaded is not None
    assert loaded.to_dict() == rec.to_dict()


def test_history_dedup(tmp_path):
    rec1 = ProcessingRecord(processing_id="PSR-2026-0001")
    history = ProcessingHistory(directory=tmp_path)
    history.add(rec1)
    rec2 = ProcessingRecord(processing_id="PSR-2026-0001", notes="changed")
    history.add(rec2)
    assert len(history.records) == 1
    assert history.records[0].notes == "changed"


def test_id_generation():
    ids = new_processing_id(year=2026)
    assert ids.startswith("PSR-2026-")
    existing = {"PSR-2026-00001", "PSR-2026-00002"}
    assert new_processing_id(year=2026, existing=existing) == "PSR-2026-00003"