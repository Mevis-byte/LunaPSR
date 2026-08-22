"""Image / dataset I/O for LunaPSR.

Supports PNG, JPEG, TIFF and raw numeric arrays.  16-bit and float scientific
rasters are kept in their native scale; only the *visualisation* path converts
to 8-bit.  Where the format supports it (TIFF) we try to write metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from core.types import ImageData, infer_bit_depth, to_float32_grayscale

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# OHRC metadata carried over from the browse-product filename when available.
def _parse_ohrc_name(name: str) -> dict:
    """Best-effort parse of an ISRO browse product filename like
    ``ch2_ohr_ncp_20210331T2033243734_b_brw_d18.png``.
    """
    import re

    meta: dict = {}
    base = Path(name).stem
    m = re.match(r"ch2_ohr_([a-z0-9]+)_(\d{8})T(\d{1,})_([a-z])_([a-z0-9]+)_d(\d+)", base)
    if m:
        meta["mission"] = "Chandrayaan-2"
        meta["instrument"] = "OHRC"
        meta["target_type"] = m.group(1)
        meta["acquisition_date"] = m.group(2)
        meta["acquisition_time_ut"] = m.group(3)
        meta["redundancy"] = m.group(4)
        meta["level"] = m.group(5)
        meta["sample_interval"] = int(m.group(6))
    return meta


def load_image(path) -> ImageData:
    """Load an image file into an :class:`ImageData`.

    Raises
    ------
    FileNotFoundError, ValueError
        If the file is missing or cannot be decoded as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    arr = _read_with_fallback(path)
    bit_depth = infer_bit_depth(arr)
    arr = to_float32_grayscale(arr)

    name = path.name
    metadata = _parse_ohrc_name(name)
    metadata.update(_read_text_metadata(path))

    return ImageData(
        data=arr,
        bit_depth=bit_depth,
        source_name=name,
        source_path=str(path),
        metadata=metadata,
    )


def load_array(arr: np.ndarray, name: str = "array", metadata: Optional[dict] = None) -> ImageData:
    """Build an ImageData from an in-memory array."""
    data = to_float32_grayscale(arr)
    bit_depth = infer_bit_depth(arr)
    return ImageData(
        data=data,
        bit_depth=bit_depth,
        source_name=name,
        metadata=metadata or {},
    )


def _read_with_fallback(path: Path) -> np.ndarray:
    # Try OpenCV first for robustness with wide format support.
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass
    # Fall back to PIL (handles palettes/indexed greyscale gracefully).
    from PIL import Image

    with Image.open(path) as pil:
        # Lazy-load and merge any palette to raw values.
        if pil.mode in ("P", "PO", "PA"):
            pil = pil.convert("RGBA")
        elif pil.mode == "I;16" or pil.mode == "I;16B":
            pil = pil.convert("I")
        if getattr(pil, "mode", "") not in ("L", "LA", "RGB", "RGBA", "I", "F"):
            pil = pil.convert("RGB")
        return np.asarray(pil)


def save_8bit_png(arr: np.ndarray, path, percentiles: tuple[float, float] = (0.5, 99.5)) -> Path:
    """Save an 8-bit PNG visualisation of *any* numeric image."""
    from core.preprocessing.normalization import display_stretch

    out = display_stretch(arr, percentiles=percentiles, dtype="uint8")
    from PIL import Image

    path = Path(path)
    Image.fromarray(out).save(path)
    return path


def save_16bit_tiff(arr: np.ndarray, path, metadata: Optional[dict] = None) -> Path:
    """Save a 16-bit TIFF, preserving numeric scale where possible.

    Positive values are clamped to [0, 65535].  Adds textual metadata tags when
    a writer supports them.
    """
    arr = np.asarray(arr).astype(np.float32)
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Cannot export array containing NaN/Inf values")
    lo, hi = float(np.min(arr)), float(np.max(arr))
    scale = 65535.0 / max(hi, 1e-12)
    scaled = np.clip((arr - lo) * scale, 0, 65535).astype(np.uint16)
    from PIL import Image

    path = Path(path)
    img = Image.fromarray(scaled)
    if metadata:
        import json

        _meta_text = json.dumps(metadata, default=str, indent=0)
        img.save(path, format="TIFF", description=_meta_text)
    else:
        img.save(path, format="TIFF")
    return path


def _read_text_metadata(path: Path) -> dict:
    """Read embedded text metadata (PNG tEXt/iTXt chunks, TIFF description).

    Keys embedded as ``LunaPSR-<key>`` are stored unprefixed (``LunaPSR-mission``
    becomes ``mission``) so they merge cleanly with filename-parsed metadata.
    """
    out: dict = {}
    try:
        from PIL import Image

        with Image.open(path) as pil:
            info = getattr(pil, "info", {}) or {}
            text = getattr(pil, "text", None) or {}
            for source in (text, info):
                for k, v in source.items():
                    if not isinstance(v, str):
                        continue
                    kkey = k.strip().lower()
                    if kkey.startswith("description") or kkey == "imagedescription":
                        try:
                            import json

                            parsed = json.loads(v)
                            if isinstance(parsed, dict):
                                out.update(parsed)
                                continue
                        except Exception:
                            pass
                        out["description"] = v
                    elif kkey.startswith("lunapsr-"):
                        out[k.strip()[len("LunaPSR-"):]] = v
                    elif kkey in ("processing_id", "pipeline", "source_name", "timestamp", "mission", "instrument"):
                        out[k.strip()] = v
    except Exception:
        pass
    return out


def embed_text_metadata(path, metadata: dict) -> Path:
    """Embed metadata as PNG tEXt chunks (preserved on reload).

    Writes to a temporary file first, then replaces the target, so a failure
    never leaves a truncated image behind.
    """
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp.png")
    with Image.open(path) as pil:
        img = pil.convert("L") if pil.mode != "L" else pil
        pnginfo = PngInfo()
        pnginfo.add_text("Description", str(metadata))
        for k, v in metadata.items():
            pnginfo.add_text(f"LunaPSR-{k}", str(v))
        img.save(tmp, pnginfo=pnginfo, optimize=False)
    tmp.replace(path)
    return path


def save_numpy(arr: np.ndarray, path) -> Path:
    """Lossless raw export (numpy .npz) for downstream scientific tools."""
    np.savez_compressed(str(path), data=np.asarray(arr).astype(np.float32))
    return Path(path)