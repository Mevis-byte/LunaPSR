"""Geometry grid parsing and geolocation.

Parses the selenographic tie-point grid shipped with an OHRC product
(``Longitude,Latitude,Pixel,Scan``) and provides bilinear interpolation from
image coordinates to (longitude, latitude).  The grid is read-only; arrays are
loaded once and shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from core.ohrc.exceptions import CorruptProductError, ProductDiscoveryError


@dataclass
class GeometryGrid:
    """A regular grid of (pixel, scan) -> (lon, lat) tie-points."""

    pixels: np.ndarray
    scans: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    source: str = ""

    def _build_interpolators(self) -> None:
        from scipy.interpolate import RegularGridInterpolator

        self._lon_interp = RegularGridInterpolator(
            (self.scans, self.pixels),
            self.lon,
            bounds_error=False,
            fill_value=np.nan,
        )
        self._lat_interp = RegularGridInterpolator(
            (self.scans, self.pixels),
            self.lat,
            bounds_error=False,
            fill_value=np.nan,
        )

    def geolocate(self, pixel: float, scan: float) -> Optional[Tuple[float, float]]:
        """Return (longitude, latitude) for an image point, or ``None`` if the
        point lies outside the tie-point grid."""
        if not hasattr(self, "_lon_interp"):
            self._build_interpolators()
        pt = np.asarray([[float(scan), float(pixel)]], dtype=np.float64)
        lon = float(self._lon_interp(pt)[0])
        lat = float(self._lat_interp(pt)[0])
        if np.isnan(lon) or np.isnan(lat):
            return None
        return (lon, lat)

    @property
    def bounds(self) -> dict:
        return {
            "pixel_min": int(self.pixels.min()),
            "pixel_max": int(self.pixels.max()),
            "scan_min": int(self.scans.min()),
            "scan_max": int(self.scans.max()),
            "lon_min": float(np.nanmin(self.lon)),
            "lon_max": float(np.nanmax(self.lon)),
            "lat_min": float(np.nanmin(self.lat)),
            "lat_max": float(np.nanmax(self.lat)),
        }


def parse_geometry_csv(csv_path: Path) -> GeometryGrid:
    """Parse a ``Longitude,Latitude,Pixel,Scan`` CSV grid file."""
    path = Path(csv_path)
    if not path.exists():
        raise ProductDiscoveryError(f"Geometry grid file not found: '{path}'.")

    try:
        data = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=(0, 1, 2, 3),
            dtype=np.float64,
        )
    except (OSError, ValueError) as exc:
        raise CorruptProductError(
            f"Geometry grid file '{path.name}' could not be parsed."
        ) from exc

    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 4:
        raise CorruptProductError(
            f"Geometry grid file '{path.name}' has an unexpected shape."
        )

    lon = data[:, 0]
    lat = data[:, 1]
    pixel = data[:, 2]
    scan = data[:, 3]

    order = np.lexsort((pixel, scan))  # sort by scan then pixel
    lon, lat, pixel, scan = lon[order], lat[order], pixel[order], scan[order]

    unique_pixels = np.unique(pixel)
    unique_scans = np.unique(scan)
    expected = len(unique_pixels) * len(unique_scans)
    if expected != pixel.size:
        raise CorruptProductError(
            f"Geometry grid file '{path.name}' is not a regular grid "
            f"({pixel.size} points but expected {expected} for "
            f"{len(unique_pixels)} pixels x {len(unique_scans)} scans)."
        )

    n_pixels = len(unique_pixels)
    n_scans = len(unique_scans)
    return GeometryGrid(
        pixels=np.asarray(unique_pixels, dtype=np.float64),
        scans=np.asarray(unique_scans, dtype=np.float64),
        lon=lon.reshape(n_scans, n_pixels),
        lat=lat.reshape(n_scans, n_pixels),
        source=str(path),
    )