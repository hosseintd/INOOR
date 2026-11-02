"""
Regression tests for scaling frames to a common level before stacking.

Historical behaviour: create_master combined frames with a plain median and
no per-frame scaling. Sky flats fade while the sequence is being taken, so
the per-pixel median simply picked whichever frame happened to sit in the
middle and the combination carried ONE frame's noise. Measured on 5 real
g-band twilight flats (medians 3046 -> 2668 ADU, a 13.4% fade), high-pass
noise in a central 1000x1000 patch was 57.4 ADU for a single frame, 50.3 for
the unscaled median of five, and 33.4 once the frames were put on a common
level.

Scaling is WRONG for bias and dark, whose absolute level is the signal, so it
is opt-in, defaults off, and is always recorded in the header.
"""
import numpy as np
import pytest
from astropy.io import fits

from core import masterFrame_creator as mfc


def write_fits(path, data):
    fits.PrimaryHDU(np.asarray(data, dtype='float32')).writeto(str(path))
    return str(path)


def fading_flats(tmp_path, levels, shape=(64, 64), seed=0):
    """One fixed illumination shape seen at several brightness levels."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    r2 = ((xx - cx) / shape[1]) ** 2 + ((yy - cy) / shape[0]) ** 2
    shape_map = 1.0 - 0.6 * r2  # gentle vignetting, min 0.7 at the corners
    paths = []
    for i, lvl in enumerate(levels):
        frame = lvl * shape_map + rng.normal(0, 0.02 * lvl, shape)
        paths.append(write_fits(tmp_path / f"flat_{i}.fits", frame))
    return paths, shape_map


def test_off_by_default_leaves_stacking_unchanged(tmp_path):
    paths, _ = fading_flats(tmp_path, [3000.0, 2900.0, 2800.0])
    master, hdr = mfc.create_master(None, paths, method='median')
    plain = np.median(np.dstack([mfc.load_fits(p) for p in paths]), axis=2)
    assert np.allclose(master, plain)
    assert hdr['STK_SCAL'] is False
    assert 'STK_SREF' not in hdr


def test_scaling_recovers_the_common_shape(tmp_path):
    """A pure level difference must not survive into the master's shape."""
    levels = [3046.0, 2918.0, 2815.0, 2717.0, 2668.0]
    paths, shape_map = fading_flats(tmp_path, levels)

    scaled, hdr = mfc.create_master(None, paths, method='median',
                                    scale_to_common_level=True)
    assert hdr['STK_SCAL'] is True
    # STK_SREF is the median of the frame medians, so it carries the
    # illumination shape's own median, not the bare brightness level.
    expected_ref = np.median(levels) * np.median(shape_map)
    assert hdr['STK_SREF'] == pytest.approx(expected_ref, rel=0.02)

    # Normalised, the master must reproduce the illumination shape. The
    # tolerance is set by the per-pixel noise in the fixture, not by the
    # scaling, so compare scatter rather than the worst single pixel.
    got = scaled / np.median(scaled)
    want = shape_map / np.median(shape_map)
    assert np.std(got - want) < 0.015
    assert np.abs(np.mean(got - want)) < 0.002


def test_scaling_beats_the_plain_median_on_noise(tmp_path):
    """The whole point: five fading flats should behave like five, not one."""
    levels = [3046.0, 2918.0, 2815.0, 2717.0, 2668.0]
    paths, shape_map = fading_flats(tmp_path, levels, shape=(256, 256), seed=7)

    plain, _ = mfc.create_master(None, paths, method='median')
    scaled, _ = mfc.create_master(None, paths, method='median',
                                  scale_to_common_level=True)

    def residual_rms(master):
        norm = master / np.median(master)
        want = shape_map / np.median(shape_map)
        return float(np.std(norm - want))

    # On the real 5-flat g-band set the scaled combination came out at 0.66x
    # the noise of the unscaled one (33.4 vs 50.3 ADU); this fixture
    # reproduces that ratio closely.
    assert residual_rms(scaled) < 0.75 * residual_rms(plain)


def test_refuses_to_scale_a_frame_with_a_non_positive_level(tmp_path):
    """Loud failure: a bias-like frame around zero must not be scaled."""
    good = write_fits(tmp_path / "a.fits", np.full((32, 32), 100.0))
    zero = write_fits(tmp_path / "b.fits", np.zeros((32, 32)))
    with pytest.raises(ValueError, match="common level"):
        mfc.create_master(None, [good, zero], method='median',
                          scale_to_common_level=True)


def test_scaling_is_recorded_even_when_off(tmp_path):
    paths, _ = fading_flats(tmp_path, [100.0, 110.0])
    _, hdr = mfc.create_master(None, paths, method='median')
    assert 'STK_SCAL' in hdr, "provenance must say whether scaling ran"
