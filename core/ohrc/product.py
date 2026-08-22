"""OHRCProduct facades and the product load lifecycle.

``OHRCProduct.load(product_path)`` is the single entry point for ingestion:

    product = OHRCProduct.load(path)
    product.image          -> OHRCImageData (scientific array, dtype, dims, units)
    product.metadata       -> parsed PDS4 metadata
    product.geometry       -> GeometryGrid (or None)
    product.product_info   -> plain dict summary for the UI

Loading is read-only.  The scientific array is memory-mapped (never copied),
so a 1.2 GB product uses ~200 MB of address space and does not duplicate the
data.  Nothing here modifies any file inside the product.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from core.ohrc.exceptions import OHRCError
from core.ohrc.geometry import GeometryGrid, parse_geometry_csv
from core.ohrc.reader import discover_product, open_scientific_array, parse_label_file
from core.ohrc.reader import ProductLayout


@dataclass
class OHRCImageData:
    """The scientific image in memory (read-only, no copy)."""

    array: np.ndarray
    dtype: np.dtype
    bit_depth: int
    byte_order: Optional[str]
    signed: bool
    pds_datatype: str
    dimensions: tuple
    samples: int
    lines: int
    units: Optional[str]
    units_comment: Optional[str]
    source_path: Optional[Path]
    label_md5: Optional[str]

    def scientific_array(self) -> np.ndarray:
        """The original scientific values, untouched."""
        return self.array

    def value(self, sample: int, line: int) -> Optional[float]:
        """Read a single pixel.  ``sample`` = X, ``line`` = Y."""
        if not (0 <= sample < self.samples and 0 <= line < self.lines):
            return None
        return float(self.array[line, sample])


class OHRCProduct:
    """A loaded Chandrayaan-2 OHRC scientific product."""

    def __init__(
        self,
        path,
        layout: ProductLayout,
        metadata,
        array: np.ndarray,
        geometry: Optional[GeometryGrid] = None,
    ) -> None:
        self.path = Path(path)
        self.layout = layout
        self.metadata = metadata
        self._array = array
        self.geometry = geometry
        self.browse_summary: Optional[dict] = None

    # -- facades ----------------------------------------------------------
    @property
    def image(self) -> OHRCImageData:
        dt = self.metadata.array.data_type
        arr = self._array
        return OHRCImageData(
            array=arr,
            dtype=arr.dtype,
            bit_depth=dt.bit_depth,
            byte_order=dt.byte_order,
            signed=dt.signed,
            pds_datatype=dt.pds_type,
            dimensions=arr.shape,
            samples=int(arr.shape[1]),
            lines=int(arr.shape[0]),
            units=self.metadata.calibration.units,
            units_comment=self.metadata.calibration.units_comment,
            source_path=self.layout.scientific_image,
            label_md5=self.metadata.source_file.md5_checksum,
        )

    @property
    def product_info(self) -> dict:
        img = self.image
        m = self.metadata
        return {
            "product_name": m.product_name,
            "product_path": str(self.path),
            "product_type": m.product_class,
            "processing_level": m.processing_level,
            "processing_level_description": m.calibration.level_description,
            "dimensions": {"lines": img.lines, "samples": img.samples,
                           "total_pixels": int(img.lines * img.samples)},
            "datatype": img.pds_datatype,
            "bit_depth": img.bit_depth,
            "signed": img.signed,
            "byte_order": img.byte_order,
            "units": img.units,
            "units_comment": img.units_comment,
            "acquisition": {
                "start_date_time": m.acquisition.start_date_time,
                "stop_date_time": m.acquisition.stop_date_time,
                "imaging_orbit_number": m.acquisition.imaging_orbit_number,
                "dumping_orbit_number": m.acquisition.dumping_orbit_number,
                "spacecraft_altitude_km": m.acquisition.spacecraft_altitude_km,
                "target": m.acquisition.target,
                "area": m.acquisition.area,
            },
            "camera": {
                "name": m.camera.name,
                "band": m.camera.band,
                "focal_length_mm": m.camera.focal_length_mm,
                "detector_pixel_width_um": m.camera.detector_pixel_width_um,
                "pixel_resolution_mp": m.camera.pixel_resolution_mp,
            },
            "observation_geometry": {
                "roll_deg": m.geometry.roll_deg,
                "pitch_deg": m.geometry.pitch_deg,
                "yaw_deg": m.geometry.yaw_deg,
                "sun_azimuth_deg": m.geometry.sun_azimuth_deg,
                "sun_elevation_deg": m.geometry.sun_elevation_deg,
                "solar_incidence_deg": m.geometry.solar_incidence_deg,
                "projection": m.geometry.projection,
                "corners": m.geometry.corners,
            },
            "calibration": {
                "coefficients": m.calibration.coefficients,
                "calibration_note": m.calibration.calibration_note,
            },
            "files": {
                "scientific_image": str(self.layout.scientific_image),
                "scientific_label": str(self.layout.scientific_label)
                if self.layout.scientific_label else None,
                "browse_image": str(self.layout.browse_image)
                if self.layout.browse_image else None,
                "geometry_grid": str(self.layout.geometry_grid)
                if self.layout.geometry_grid else None,
                "md5_checksum": m.source_file.md5_checksum,
            },
            "metadata": m.to_dict(),
        }

    # -- loading ----------------------------------------------------------
    @classmethod
    def load(cls, product_path, load_geometry: bool = True) -> "OHRCProduct":
        """Load a product: discover, parse metadata, validate, memory-map.

        Raises an ``OHRCError`` subclass with a user-facing message whenever
        the product cannot be ingested safely.
        """
        layout = discover_product(product_path)
        metadata = parse_label_file(layout.scientific_label)
        array = open_scientific_array(layout.scientific_image, metadata.array)

        geometry = None
        if load_geometry and layout.geometry_grid is not None:
            try:
                geometry = parse_geometry_csv(layout.geometry_grid)
            except OHRCError:
                geometry = None  # geometry is auxiliary; scientific load is unaffected

        product = cls(
            path=layout.root,
            layout=layout,
            metadata=metadata,
            array=array,
            geometry=geometry,
        )
        product._try_browse_summary()
        return product

    # -- helpers ----------------------------------------------------------
    def decimate_average(self, factor: int = 10) -> np.ndarray:
        """Block-average decimation of scientific values (float)."""
        from core.ohrc.display import block_average

        return block_average(self._array, factor)

    def decimate_stride(self, factor: int = 10) -> np.ndarray:
        """Raw stride decimation (distribution-preserving)."""
        arr = np.asarray(self._array)
        return arr[0::factor, 0::factor]

    @property
    def source_md5_verified(self) -> Optional[bool]:
        """Whether the on-disk MD5 matches the label (None = not computed)."""
        return None

    def verify_md5(self) -> bool:
        """Compute the on-disk MD5 and compare with the label (full read)."""
        import hashlib

        expected = self.metadata.source_file.md5_checksum
        if not expected:
            return False
        h = hashlib.md5()
        with open(self.layout.scientific_image, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest() == expected

    def _try_browse_summary(self) -> None:
        if self.layout.browse_image is None:
            self.browse_summary = None
            return
        try:
            from PIL import Image

            with Image.open(self.layout.browse_image) as im:
                self.browse_summary = {
                    "path": str(self.layout.browse_image),
                    "width": im.width,
                    "height": im.height,
                    "mode": im.mode,
                }
        except OSError:
            self.browse_summary = None