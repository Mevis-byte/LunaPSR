"""Product discovery + raw scientific-image loading.

Discovery locates the pieces of an OHRC bundle on disk without modifying
anything.  Loading maps the scientific ``.img`` as a read-only memory-mapped
numpy array and rejects malformed products before any element is touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.ohrc.exceptions import (
    CorruptProductError,
    DimensionMismatchError,
    MetadataNotFoundError,
    ProductDiscoveryError,
    ScientificImageNotFoundError,
)
from core.ohrc.metadata import ArrayInfo, OHRCMetadata, parse_scientific_label

_SCIENTIFIC_EXTENSIONS = {".img"}
_BROWSE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_GEOMETRY_EXTENSIONS = {".csv"}


def _walk(root: Path):
    return sorted([p for p in root.rglob("*") if p.is_file()])


@dataclass
class ProductLayout:
    """Everything discovered inside one OHRC product directory."""

    root: Path
    product_name: str
    scientific_image: Optional[Path] = None
    scientific_label: Optional[Path] = None
    browse_image: Optional[Path] = None
    browse_label: Optional[Path] = None
    geometry_grid: Optional[Path] = None
    geometry_label: Optional[Path] = None
    auxiliary_files: List[Path] = field(default_factory=list)
    all_files: List[Path] = field(default_factory=list)

    @property
    def complete_science_chain(self) -> bool:
        return (
            self.scientific_image is not None and self.scientific_label is not None
        )


def discover_product(product_path) -> ProductLayout:
    """Locate the science image, its label, browse and geometry files.

    Raises ``ProductDiscoveryError`` when the directory cannot be used,
    ``ScientificImageNotFoundError`` when no scientific image is found, and
    ``MetadataNotFoundError`` when the science image has no label.
    """
    root = Path(product_path)
    if not root.exists():
        raise ProductDiscoveryError(
            f"OHRC product path not found: '{root}'"
        )
    if not root.is_dir():
        raise ProductDiscoveryError(
            f"OHRC product path is not a directory: '{root}'"
        )

    files = _walk(root)
    if not files:
        raise ProductDiscoveryError(f"OHRC product directory is empty: '{root}'")

    layout = ProductLayout(root=root, product_name=root.name, all_files=files)

    # Science image: any *.img under the bundle; prefer a data/ sibling label.
    candidates = [f for f in files if f.suffix.lower() in _SCIENTIFIC_EXTENSIONS]
    if len(candidates) > 1:
        priority = [c for c in candidates if "data" in c.parts]
        if len(priority) == 1:
            candidates = priority
        else:
            raise ProductDiscoveryError(
                f"Multiple scientific images found in '{root}': "
                + ", ".join(str(c.name) for c in candidates)
            )
    if not candidates:
        raise ScientificImageNotFoundError(
            f"Scientific image file not found (no *.img in '{root}')."
        )

    layout.scientific_image = candidates[0]
    layout.scientific_label = _sibling_label(layout.scientific_image)
    if layout.scientific_label is None:
        raise MetadataNotFoundError(
            f"OHRC metadata file not found for scientific image "
            f"'{layout.scientific_image.name}'."
        )

    for f in files:
        suffix = f.suffix.lower()
        if suffix in _BROWSE_EXTENSIONS and _is_browse_name(f):
            layout.browse_image = f
            layout.browse_label = _sibling_label(f)
        elif suffix in _GEOMETRY_EXTENSIONS and _is_geometry_name(f):
            layout.geometry_grid = f
            layout.geometry_label = _sibling_label(f)
        else:
            layout.auxiliary_files.append(f)

    if layout.browse_image is None:
        pngs = [f for f in files if f.suffix.lower() in _BROWSE_EXTENSIONS]
        if len(pngs) == 1:
            layout.browse_image = pngs[0]
            layout.browse_label = _sibling_label(pngs[0])

    if layout.geometry_grid is None:
        csvs = [f for f in files if f.suffix.lower() in _GEOMETRY_EXTENSIONS]
        if len(csvs) == 1:
            layout.geometry_grid = csvs[0]
            layout.geometry_label = _sibling_label(csvs[0])

    layout.auxiliary_files = [f for f in layout.auxiliary_files
                              if f.suffix.lower() not in _BROWSE_EXTENSIONS
                              and f.suffix.lower() not in _GEOMETRY_EXTENSIONS]
    return layout


def _sibling_label(file_path: Path) -> Optional[Path]:
    candidate = file_path.with_suffix(".xml")
    if candidate.exists():
        return candidate
    return None


def _is_browse_name(path: Path) -> bool:
    return any(token in path.name.lower() for token in ("_b_", "_brw", "browse"))


def _is_geometry_name(path: Path) -> bool:
    return any(token in path.name.lower() for token in ("_g_", "_grd", "geometry"))


def validate_file_size(img_path: Path, metadata: OHRCMetadata) -> None:
    """Ensure the on-disk byte count matches the label before mapping."""
    expected = metadata.array.expected_bytes()
    offset = int(metadata.array.offset_bytes or 0)
    actual = img_path.stat().st_size
    if expected is None:
        raise MetadataNotFoundError(
            f"OHRC metadata for '{img_path.name}' does not declare a supported "
            "array size (dimensions or datatype missing)."
        )
    if actual != offset + expected:
        raise DimensionMismatchError(
            f"Image dimensions do not match the data file '{img_path.name}': "
            f"label declares {expected} bytes at offset {offset} "
            f"({metadata.array.lines} x {metadata.array.samples} x "
            f"{np.dtype(metadata.array.data_type.numpy_dtype).itemsize} bytes), "
            f"but the file contains {actual} bytes. "
            "The product appears corrupted or truncated; refusing to load."
        )


def array_from_bytes(data: bytes, array_info: ArrayInfo) -> np.ndarray:
    """Build an in-memory array from raw bytes (used by small unit tests)."""
    dtype = np.dtype(array_info.data_type.numpy_dtype)
    expected = array_info.expected_bytes()
    if len(data) < expected:
        raise DimensionMismatchError(
            f"Image data is truncated: expected {expected} bytes, "
            f"received {len(data)} bytes."
        )
    if offset := array_info.offset_bytes:
        data = data[offset:]
    arr = np.frombuffer(data, dtype=dtype, count=array_info.total_pixels)
    return arr.reshape(array_info.shape).copy()


def open_scientific_array(img_path: Path, array_info: ArrayInfo) -> np.ndarray:
    """Open the scientific image as a read-only memory-map.

    The array is validated against the label *before* mapping: size, offset,
    and dtype are checked so that truncation or a malformed product surface as
    errors instead of a silently mis-shaped array.
    """
    dtype = np.dtype(array_info.data_type.numpy_dtype)
    shape = array_info.shape
    offset = int(array_info.offset_bytes or 0)

    if shape is None:
        raise MetadataNotFoundError(
            "OHRC metadata does not declare valid dimensions for the image."
        )
    if dtype is None:
        raise MetadataNotFoundError(
            "OHRC metadata does not declare a supported datatype."
        )

    expected = int(np.prod(shape) * dtype.itemsize)
    actual = img_path.stat().st_size
    if actual != offset + expected:
        raise DimensionMismatchError(
            f"Image dimensions do not match the data file '{img_path.name}': "
            f"label declares {expected} bytes at offset {offset} "
            f"({shape[0]} x {shape[1]} x {dtype.itemsize} bytes), "
            f"file contains {actual} bytes. "
            "The product appears corrupted or truncated; refusing to load."
        )
    if offset + expected > actual:
        raise CorruptProductError(
            f"Declared array offset {offset} exceeds the data file size."
        )

    return np.memmap(img_path, mode="r", dtype=dtype, offset=offset, shape=shape)


def parse_label_file(label_path: Path) -> OHRCMetadata:
    """Read and parse a scientific-image PDS4 label."""
    try:
        xml_bytes = label_path.read_bytes()
    except OSError as exc:
        raise MetadataNotFoundError(
            f"OHRC metadata file not found / unreadable: '{label_path.name}'."
        ) from exc
    metadata = parse_scientific_label(xml_bytes)
    return metadata