import numpy as np
import pytest

from core.pipeline import EnhancementPipeline
from core.registry import validate_stage


def test_empty_pipeline_is_identity():
    img = np.random.rand(32, 32).astype(np.float32)
    p = EnhancementPipeline()
    res = p.run(img)
    assert np.allclose(res.image, img)


def test_pipeline_stages_recorded():
    p = EnhancementPipeline(name="t")
    p.add("normalize", percentiles=(1, 99))
    p.add("gamma", gamma=0.7)
    p.add("clahe", clip_limit=2.0, tile_size=8)
    md = p.stages_metadata()
    assert [m["stage"] for m in md] == ["normalize", "gamma", "clahe"]
    assert md[1]["parameters"]["gamma"] == 0.7


def test_pipeline_reproducible(noisy_image):
    p = EnhancementPipeline()
    p.add("normalize")
    p.add("gamma", gamma=0.8)
    p.add("clahe", clip_limit=1.5, tile_size=8)
    a = p.run(noisy_image * 255.0)
    b = p.run(noisy_image * 255.0)
    assert np.array_equal(a.image, b.image)
    assert a.stage_metadata_list() == b.stage_metadata_list()


def test_recreated_pipeline_same_result(noisy_image):
    p1 = EnhancementPipeline()
    p1.add("normalize")
    p1.add("gamma", gamma=0.75)
    r1 = p1.run(noisy_image * 1000.0)

    p2 = EnhancementPipeline()
    p2.from_dict(p1.to_dict())
    r2 = p2.run(noisy_image * 1000.0)
    assert np.array_equal(r1.image, r2.image)


def test_unknown_stage_rejected():
    p = EnhancementPipeline()
    with pytest.raises(KeyError):
        p.add("no_such_stage")


def test_validate_stage():
    validate_stage("clahe")
    validate_stage("retinex")
    with pytest.raises(KeyError):
        validate_stage("nope")


def test_nan_input_rejected():
    p = EnhancementPipeline()
    p.add("normalize")
    bad = np.zeros((16, 16))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        p.run(bad)


def test_run_bad_shape():
    p = EnhancementPipeline()
    with pytest.raises(ValueError):
        p.run(np.zeros(10))


def test_16bit_input_pipeline(sample_ohrc):
    """Run the recommended pipeline on the sample OHRC data end-to-end."""
    p = EnhancementPipeline()
    p.add("normalize")
    p.add("gamma", gamma=0.75)
    p.add("clahe", clip_limit=2.0, tile_size=8)
    res = p.run(sample_ohrc.data)
    assert res.image.shape == sample_ohrc.shape
    assert np.isfinite(res.image).all()
    assert res.image.min() >= 0.0 and res.image.max() <= 1.0
    assert res.normalization["stage"] == "normalize"