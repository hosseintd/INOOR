"""
Regression tests for the unified master-frame stacking path.

Historical bug: CalibrationWorker.make_master_or_none used a completely
different combination algorithm when frames needed binning (per-frame global
sigma-clip + plain np.median) than when they didn't (NaN-safe nanmedian +
post-stack 2D block-median clip). Whether outlier rejection ran -- and how --
silently depended on incidental file dimensions.

Now everything goes through calibration.create_master_from_list ->
masterFrame_creator.create_master, with an optional frame_transform for
binning. These tests pin that behavior:

  1. Binning path and direct path produce IDENTICAL masters when fed
     equivalent data.
  2. Stacking is NaN-safe (a NaN pixel in one frame must not corrupt the
     combined master at that pixel).
  3. The 2D sigma-clip post-filter runs identically on both paths.
"""
import numpy as np
import pytest
from astropy.io import fits

from core import calibration as calib


def write_fits(path, data):
    fits.PrimaryHDU(np.asarray(data, dtype='float32')).writeto(str(path))
    return str(path)


def bin2(img):
    """2x2 average binning identical to CalibrationWorker.bin_image."""
    ny, nx = img.shape[0] // 2, img.shape[1] // 2
    return img[:ny * 2, :nx * 2].reshape(ny, 2, nx, 2).mean(axis=(1, 3))


def make_frames(tmp_path, values, shape=(8, 8), upsample=1, prefix="f"):
    """Write constant-valued frames; upsample>1 writes each frame at
    (shape*upsample) so that average-binning reproduces the base frame."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, v in enumerate(values):
        base = np.full(shape, v, dtype='float32')
        if upsample > 1:
            big = np.kron(base, np.ones((upsample, upsample), dtype='float32'))
            paths.append(write_fits(tmp_path / f"{prefix}{i}_big.fits", big))
        else:
            paths.append(write_fits(tmp_path / f"{prefix}{i}.fits", base))
    return paths


@pytest.mark.parametrize("method", ["median", "mean"])
@pytest.mark.parametrize("do_clip", [False, True])
def test_binned_and_direct_paths_produce_identical_masters(tmp_path, method, do_clip):
    values = [10.0, 12.0, 14.0, 100.0]  # 100 = outlier frame

    direct_paths = make_frames(tmp_path / "a", values, prefix="d")
    binned_paths = make_frames(tmp_path / "b", values, upsample=2, prefix="u")

    master_direct = calib.create_master_from_list(
        direct_paths, method=method, do_sigma_clip=do_clip)

    master_binned = calib.create_master_from_list(
        binned_paths, method=method, do_sigma_clip=do_clip,
        frame_transform=lambda img: bin2(img) if img.shape != (8, 8) else img)

    assert master_direct.shape == master_binned.shape == (8, 8)
    np.testing.assert_allclose(master_binned, master_direct, rtol=1e-6,
                               err_msg="Binned and direct stacking paths diverged -- "
                                       "the combination algorithm depends on frame shape again")


def test_stacking_is_nan_safe(tmp_path):
    f1 = np.full((8, 8), 10.0, dtype='float32')
    f2 = np.full((8, 8), 12.0, dtype='float32')
    f3 = np.full((8, 8), 14.0, dtype='float32')
    f2[3, 3] = np.nan  # dead/masked pixel in one frame

    paths = [write_fits(tmp_path / f"n{i}.fits", f) for i, f in enumerate((f1, f2, f3))]
    master = calib.create_master_from_list(paths, method='median')

    assert np.isfinite(master[3, 3]), "NaN in a single frame corrupted the master"
    # nanmedian of (10, 14) at the NaN pixel
    assert master[3, 3] == pytest.approx(12.0)
    # elsewhere: median of (10, 12, 14)
    assert master[0, 0] == pytest.approx(12.0)


def _reference_2d_sigma_clip(data, sigma_lower, sigma_upper, kernel_size, auto_sigma=False):
    """The original per-block astropy implementation, kept here as the spec the
    vectorised apply_2d_sigma_clip must reproduce exactly."""
    from astropy.stats import sigma_clip

    ny, nx = data.shape
    processed = data.copy()
    ks = max(3, kernel_size)
    slo = 100.0 if auto_sigma else sigma_lower
    shi = sigma_upper

    for y in range(0, ny, ks):
        for x in range(0, nx, ks):
            block = data[y:min(y + ks, ny), x:min(x + ks, nx)]
            if block.size == 0:
                continue
            clipped = sigma_clip(block, sigma_lower=slo, sigma_upper=shi, maxiters=3)
            if hasattr(clipped, 'mask') and np.any(clipped.mask):
                processed[y:y + ks, x:x + ks] = clipped.filled(np.nanmedian(block))
    return processed


@pytest.mark.parametrize("shape,ks,slo,shi,auto", [
    ((120, 97), 5, 3.0, 3.0, False),   # ragged edge blocks
    ((64, 64), 3, 3.0, 3.0, False),
    ((100, 100), 7, 3.0, 2.5, True),   # auto mode: lower side disabled
])
def test_vectorised_2d_sigma_clip_matches_reference(shape, ks, slo, shi, auto):
    from core import masterFrame_creator as mfc

    rng = np.random.default_rng(7)
    img = rng.normal(1000, 10, shape).astype('float32')
    img[rng.integers(0, shape[0], 40), rng.integers(0, shape[1], 40)] += 500  # spikes

    np.testing.assert_allclose(
        mfc.apply_2d_sigma_clip(img, slo, shi, ks, auto_sigma=auto),
        _reference_2d_sigma_clip(img, slo, shi, ks, auto_sigma=auto),
        rtol=1e-5, atol=1e-4,
        err_msg="Vectorised block sigma-clip diverged from the astropy per-block loop")


def test_vectorised_2d_sigma_clip_handles_nan():
    from core import masterFrame_creator as mfc

    rng = np.random.default_rng(11)
    img = rng.normal(1000, 10, (64, 64)).astype('float32')
    img[rng.integers(0, 64, 20), rng.integers(0, 64, 20)] = np.nan

    out = mfc.apply_2d_sigma_clip(img, 3.0, 3.0, 5)
    ref = _reference_2d_sigma_clip(img, 3.0, 3.0, 5)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-4, equal_nan=True)
    assert np.isfinite(out).all(), "NaN pixels must be replaced by the block median"


def test_single_file_respects_frame_transform(tmp_path):
    big = np.kron(np.full((4, 4), 8.0, dtype='float32'), np.ones((2, 2), dtype='float32'))
    p = write_fits(tmp_path / "single.fits", big)
    out = calib.create_master_from_list([p], frame_transform=bin2)
    assert out.shape == (4, 4)
    assert out[0, 0] == pytest.approx(8.0)


def test_infinity_in_one_frame_does_not_poison_the_master(tmp_path):
    """
    A single +/-inf must be dropped, not averaged in.

    The guard used to be `np.isnan(stack).any()`, which does not see an
    infinity, so a stack containing one took the fast np.mean path and the
    whole mean master came out inf. Switching the guard to isfinite is not
    enough on its own either: np.nanmean([1, inf, 1]) is still inf, because
    nanmean drops NaN but not infinities. Non-finite values are mapped to NaN
    before the reduction.
    """
    good = np.full((8, 8), 5.0, dtype='float32')
    bad = good.copy()
    bad[3, 3] = np.inf
    worse = good.copy()
    worse[4, 4] = -np.inf

    paths = [write_fits(tmp_path / f"inf{i}.fits", f)
             for i, f in enumerate((good, bad, worse))]

    for method in ('mean', 'median'):
        master = calib.create_master_from_list(paths, method=method)
        assert np.all(np.isfinite(master)), f"{method} master holds a non-finite pixel"
        assert master[3, 3] == pytest.approx(5.0)
        assert master[4, 4] == pytest.approx(5.0)
