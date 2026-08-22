import numpy as np
import pytest

from core.io import load_array, load_image
from core.types import ImageData, to_float32_grayscale
from core.preprocessing.normalization import display_stretch, robust_normalize


def test_load_grayscale_png(tmp_path):
    from PIL import Image

    arr = np.linspace(0, 255, 100 * 80).reshape(100, 80).astype(np.uint8)
    p = tmp_path / "g.png"
    Image.fromarray(arr).save(p)
    img = load_image(p)
    assert img.shape == (100, 80)
    assert img.bit_depth == 8
    assert img.data.dtype == np.float32


def test_load_rgb_png(tmp_path):
    from PIL import Image

    rgb = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)
    p = tmp_path / "c.png"
    Image.fromarray(rgb).save(p)
    img = load_image(p)
    assert img.data.ndim == 2
    assert img.data.shape == (40, 50)


def test_load_16bit_tiff(tmp_path):
    arr = np.random.randint(0, 65535, (60, 60), dtype=np.uint16)
    p = tmp_path / "sixteen.tiff"
    from PIL import Image

    Image.fromarray(arr).save(p)
    img = load_image(p)
    assert img.data.max() > 255  # not collapsed to 8-bit
    assert img.bit_depth >= 16


def test_load_jpg(tmp_path):
    from PIL import Image

    p = tmp_path / "j.jpg"
    Image.fromarray(np.full((30, 30, 3), 120, dtype=np.uint8)).save(p)
    img = load_image(p)
    assert img.data.shape == (30, 30)


def test_load_unsupported_extension(tmp_path):
    p = tmp_path / "bad.gif"
    p.write_bytes(b"1")
    with pytest.raises(ValueError):
        load_image(p)


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "none.png")


def test_load_garbage(tmp_path):
    p = tmp_path / "junk.png"
    p.write_bytes(b"definitely not a png")
    with pytest.raises(Exception):
        load_image(p)


def test_rgb_luma():
    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.float32)
    g = to_float32_grayscale(rgb)
    expected = 0.299 * 10 + 0.587 * 20 + 0.114 * 30
    assert np.isclose(g[0, 0], expected, atol=1e-3)


def test_rgba_takes_alpha_away():
    rgba = np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8)
    g = to_float32_grayscale(rgba)
    assert g.ndim == 2


def test_1d_rejected():
    with pytest.raises(ValueError):
        to_float32_grayscale(np.zeros(10))


def test_invalid_shape_data():
    with pytest.raises(ValueError):
        ImageData(np.zeros((5, 5, 5, 1)))


def test_nan_rejected():
    bad = np.zeros((8, 8))
    bad[3, 3] = np.nan
    with pytest.raises(ValueError):
        ImageData(bad)


def test_ohrc_filename_parsing(sample_ohrc):
    assert sample_ohrc.metadata.get("instrument") == "OHRC"
    assert sample_ohrc.metadata.get("mission") == "Chandrayaan-2"


def test_normalization_range(sample_ohrc):
    n = robust_normalize(sample_ohrc.data)
    assert n.min() >= 0.0
    assert n.max() <= 1.0
    assert np.isfinite(n).all()


def test_display_stretch_8bit(sample_ohrc):
    d = display_stretch(sample_ohrc.data, dtype="uint8")
    assert d.dtype == np.uint8
    assert d.min() >= 0 and d.max() <= 255


def test_16bit_display_stretch(image16):
    d = display_stretch(image16.astype(np.float32), dtype="uint16")
    assert d.dtype == np.uint16
    assert d.max() <= 65535