import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np.random.seed(0)

# ---------------------------------------------------------------------------
# OHRC synthetic-product factory (Phase 2)
# ---------------------------------------------------------------------------


def _ohrc_label(bundle_name: str, data_type: str, lines: int, samples: int,
                file_name: str, file_size: int, offset: int = 0,
                endian: str = "msb", level: str = "Calibrated") -> str:
    """A minimal PDS4-style Product_Observational label resembling the real
    ISRO/ISDA OHRC label (namespace-agnostic parser must handle it)."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        '<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1" ',
        'xmlns:isda="https://isda.issdc.gov.in/pds4/isda/v1">',
        "<Identification_Area>",
        "<logical_identifier>urn:isro:isda:ch2_cho.ohr:data_calibrated:"
        + bundle_name + "</logical_identifier>",
        "<product_class>Product_Observational</product_class>",
        "</Identification_Area>",
        "<Observation_Area>",
        "<Time_Coordinates>",
        "<start_date_time>2026-01-03T10:05:17.6450Z</start_date_time>",
        "<stop_date_time>2026-01-03T10:05:33.4482Z</stop_date_time>",
        "</Time_Coordinates>",
        "<Primary_Result_Summary>",
        "<processing_level>" + level + "</processing_level>",
        "</Primary_Result_Summary>",
        "<Observing_System>",
        "<Observing_System_Component>",
        "<name>orbiter high resolution camera</name>",
        "<type>Instrument</type>",
        "</Observing_System_Component>",
        "</Observing_System>",
        "<Mission_Area>",
        '<isda:Product_Parameters>',
        "<isda:imaging_orbit_number>28372</isda:imaging_orbit_number>",
        "<isda:area>South Pole</isda:area>",
        '<isda:solar_incidence unit="deg">85.0</isda:solar_incidence>',
        "</isda:Product_Parameters>",
        "</Mission_Area>",
        "</Observation_Area>",
        "<File_Area_Observational>",
        "<File>",
        "<file_name>" + file_name + "</file_name>",
        '<file_size unit="byte">' + str(file_size) + "</file_size>",
        "</File>",
        "<Array_2D_Image>",
        '<offset unit="byte">' + str(offset) + "</offset>",
        "<axes>2</axes>",
        "<axis_index_order>Last Index Fastest</axis_index_order>",
        "<Element_Array>",
        "<data_type>" + data_type + "</data_type>",
        "</Element_Array>",
        "<Axis_Array>",
        "<axis_name>Line</axis_name>",
        "<elements>" + str(lines) + "</elements>",
        "</Axis_Array>",
        "<Axis_Array>",
        "<axis_name>Sample</axis_name>",
        "<elements>" + str(samples) + "</elements>",
        "</Axis_Array>",
        "</Array_2D_Image>",
        "</File_Area_Observational>",
        "</Product_Observational>",
    ]
    return "\n".join(parts)


_OHRC_DTYPES = {
    "UnsignedByte": np.dtype("u1"),
    "SignedByte": np.dtype("i1"),
    "UnsignedMSB2": np.dtype(">u2"),
    "UnsignedLSB2": np.dtype("<u2"),
    "SignedMSB2": np.dtype(">i2"),
    "SignedLSB2": np.dtype("<i2"),
    "SignedMSB4": np.dtype(">i4"),
    "SignedLSB4": np.dtype("<i4"),
    "UnsignedMSB4": np.dtype(">u4"),
    "UnsignedLSB4": np.dtype("<u4"),
}


def build_ohrc_bundle(root: Path, data_type: str = "UnsignedByte", lines: int = 8,
                      samples: int = 6, values=None, offset: int = 0,
                      geometry: bool = True) -> dict:
    """Create a small OHRC-like product tree under ``root`` and return its
    paths and the original ``values`` (encoded with the requested datatype)."""
    bundle = Path(root) / f"ch2_ohr_test_{lines}x{samples}"
    data_dir = bundle / "data" / "calibrated" / "20260103"
    geo_dir = bundle / "geometry" / "calibrated" / "20260103"
    data_dir.mkdir(parents=True, exist_ok=True)

    dtype = _OHRC_DTYPES.get(data_type, np.dtype("u1"))
    if values is None:
        rng = np.random.default_rng(0)
        values = rng.integers(0, 255, size=(lines, samples)).astype(np.uint8)
    encoded = np.asarray(values).astype(dtype).tobytes()

    header = b"\x00" * offset if offset else b""
    img_data = header + encoded

    img_path = data_dir / "ch2_ohr_test_d_img_d01.img"
    xml_path = data_dir / "ch2_ohr_test_d_img_d01.xml"
    img_path.write_bytes(img_data)
    xml_path.write_text(
        _ohrc_label(
            bundle.name, data_type, values.shape[0], values.shape[1],
            img_path.name, len(img_data), offset=offset,
        )
    )

    browse_dir = bundle / "browse" / "calibrated" / "20260103"
    browse_dir.mkdir(parents=True, exist_ok=True)
    png = browse_dir / "ch2_ohr_test_b_brw_d01.png"
    try:
        from PIL import Image

        Image.fromarray(np.asarray(values, dtype=np.uint8)[::2, ::2]).save(png)
    except Exception:
        png.write_bytes(b"placeholder")
    browse_dir.joinpath("ch2_ohr_test_b_brw_d01.xml").write_text(
        _ohrc_label(bundle.name, "UnsignedByte", (values.shape[0] + 1) // 2,
                    (values.shape[1] + 1) // 2, png.name, png.stat().st_size)
    )

    if geometry:
        geo_dir.mkdir(parents=True, exist_ok=True)
        csv = geo_dir / "ch2_ohr_test_g_grd_d01.csv"
        pixel_step = max(1, samples // 3)
        scan_step = max(1, lines // 3)
        pixels = sorted(set(list(range(0, samples, pixel_step)) + [samples - 1]))
        scans = sorted(set(list(range(0, lines, scan_step)) + [lines - 1]))
        rows = []
        for s in scans:
            for p in pixels:
                lon = 0.5 + p * 0.01
                lat = -80.0 - s * 0.01
                rows.append(f"{lon:.7f},{lat:.7f},{p},{s}")
        csv.write_text("Longitude,Latitude,Pixel,Scan\n" + "\n".join(rows))
        geo_dir.joinpath("ch2_ohr_test_g_grd_d01.xml").write_text(
            f"<Product><File><file_name>{csv.name}</file_name></File></Product>"
        )

    return {
        "bundle": bundle,
        "img": img_path,
        "xml": xml_path,
        "browse": png,
        "geometry_csv": bundle / "geometry" / "calibrated" / "20260103"
        / "ch2_ohr_test_g_grd_d01.csv",
        "values": values,
    }


REAL_OHRC_PRODUCT = ROOT / "ch2_ohr_ncp_20260103T1005176450_d_img_d18"


@pytest.fixture(scope="session")
def real_ohrc_product():
    """The actual Chandrayaan-2 OHRC product (Phase 2 integration)."""
    if not REAL_OHRC_PRODUCT.exists():
        pytest.skip("real OHRC product not present in project root")
    from core.ohrc import OHRCProduct

    return OHRCProduct.load(REAL_OHRC_PRODUCT)


@pytest.fixture
def noisy_image():
    """Uniform base + Gaussian noise, sigma 0.02 in [0,1] units."""
    base = np.full((256, 256), 0.5, dtype=np.float32)
    rng = np.random.default_rng(42)
    return np.clip(base + rng.normal(0, 0.02, base.shape), 0, 1).astype(np.float32)


@pytest.fixture
def gradient_image():
    """Smooth vertical gradient [0,1], no isolated noise."""
    return np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :].repeat(256, axis=0).astype(np.float32)


@pytest.fixture
def black_image():
    return np.zeros((64, 64), dtype=np.float32)


@pytest.fixture
def bright_image():
    return np.full((64, 64), 1.0, dtype=np.float32)


@pytest.fixture
def uniform_image():
    return np.full((64, 64), 0.4, dtype=np.float32)


@pytest.fixture
def tiny_image():
    return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)


@pytest.fixture
def big_image():
    return np.random.rand(2048, 2048).astype(np.float32)


@pytest.fixture
def image16():
    """Simulated 16-bit DN data (0..65535)."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 2500, size=(128, 128)).astype(np.uint16)


@pytest.fixture
def sample_ohrc():
    path = ROOT / "assets" / "sample_ohrc.png"
    if path.exists():
        from core.io import load_image

        return load_image(path)
    pytest.skip("sample OHRC image not present")


def assert_no_nan_inf(arr):
    assert np.isfinite(arr).all(), "array contains NaN/Inf"