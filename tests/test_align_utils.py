"""
Registration must recover the shift it is given, and FAIL when it cannot.

The old `ransac_align` matched stars by nearest neighbour in raw pixel
coordinates, keeping only pairs already within `match_radius` (8 px). That
presupposes the frames are aligned to 8 px, which is exactly what it was meant
to measure. Beyond ~10 px the true counterparts were all rejected and the
median was taken over chance coincidences -- and it never raised. Measured
before the fix:

    true shift dx=20  -> reported dx=1.891,  inliers=5
    true shift dx=60  -> reported dx=-1.506, inliers=2
    true shift dx=200 -> reported dx=2.429,  inliers=2

INO340 frames drift ~350 px a night, so this covered the whole working range.
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from skimage.transform import SimilarityTransform, warp

from core.align_utils import estimate_similarity


def star_field(n=150, size=1024, seed=1):
    """Synthetic field: Gaussian stars of varied brightness on flat sky."""
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, 3.0, (size, size))
    ys = rng.integers(80, size - 80, n)
    xs = rng.integers(80, size - 80, n)
    for y, x, a in zip(ys, xs, rng.uniform(300, 5000, n)):
        img[y, x] += a
    return gaussian_filter(img, 4.0)


def move(img, dx, dy, rot_deg=0.0):
    """Shift and rotate `img` about its centre; returns the moved frame."""
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    t = (SimilarityTransform(translation=(-cx, -cy))
         + SimilarityTransform(rotation=np.radians(rot_deg))
         + SimilarityTransform(translation=(cx + dx, cy + dy)))
    return warp(img, inverse_map=t.inverse, preserve_range=True, cval=100.0), t


@pytest.mark.parametrize("dx, dy", [(3, 3), (20, 20), (60, 60), (200, 200),
                                    (-350, 120)])
def test_recovers_translation(dx, dy):
    """A pure shift must come back to sub-pixel accuracy, at any size."""
    ref = star_field()
    moved, t = move(ref, dx, dy)

    tform, info = estimate_similarity(moved, ref, fwhm=8.0, threshold_sigma=5.0)

    # tform maps moved -> ref, so it must undo t.
    probe = np.array([[100.0, 100.0], [900.0, 300.0], [500.0, 800.0]])
    assert np.allclose(tform(t(probe)), probe, atol=0.5)
    assert info["rms"] < 1.0
    assert info["n_inliers"] >= 8


@pytest.mark.parametrize("rot", [0.5, 1.8])
def test_recovers_rotation(rot):
    """The INO340 field rotates ~1.8 deg/night; a translation cannot absorb it."""
    ref = star_field()
    moved, t = move(ref, 40, -25, rot_deg=rot)

    tform, info = estimate_similarity(moved, ref, fwhm=8.0, threshold_sigma=5.0)

    assert info["model"] == "similarity"
    assert abs(info["rotation_deg"] - (-rot)) < 0.05
    probe = np.array([[100.0, 100.0], [900.0, 300.0], [500.0, 800.0]])
    assert np.allclose(tform(t(probe)), probe, atol=0.5)


def test_blank_field_raises():
    """No sources means no answer -- not a plausible-looking wrong one."""
    blank = np.random.default_rng(0).normal(100.0, 3.0, (512, 512))
    with pytest.raises(ValueError):
        estimate_similarity(blank, blank.copy(), fwhm=8.0)


def test_unrelated_fields_raise():
    """Two fields with no stars in common must not produce a transform."""
    with pytest.raises(ValueError):
        estimate_similarity(star_field(seed=2, size=512),
                            star_field(seed=99, size=512),
                            fwhm=8.0, max_shift=200.0)


def test_single_bad_pair_does_not_reject_the_frame():
    """
    One mismatched pair must be rejected, not allowed to fail the frame.

    A 3*rms cut cannot do this when stars are few: on the real GRB 251013C
    g-band frame at 00:54:39 the residuals were
    [0.23 0.54 0.99 1.16 1.38 7.17] px, whose rms of 3.05 put the cut at 9.2 px
    -- so the outlier survived and the frame was rejected for the rms it had
    itself caused. The robust median+3*1.4826*MAD cut drops it.
    """
    from core.align_utils import _fit_transform

    n = 12
    rng = np.random.default_rng(3)
    src = rng.uniform(0, 1000, (n, 2))
    dst = src + np.array([25.0, -12.0]) + rng.normal(0, 0.3, (n, 2))
    dst[0] += 40.0                     # one badly mismatched pair

    tform, n_in, rms, model = _fit_transform(src, dst, max_rms=2.0)

    assert n_in == n - 1, "the outlier should have been dropped"
    assert rms < 1.0
    assert np.allclose(tform.translation, [25.0, -12.0], atol=0.5)


def test_few_stars_fall_back_to_translation():
    """
    Some real INO340 fields hold only ~5 usable stars (the GRB 251013C g-band
    Low frames give 45 detections, of which 5 are real). A rotation must not be
    invented from that; the fit drops to translation and says so.
    """
    from core.align_utils import _fit_transform

    rng = np.random.default_rng(5)
    src = rng.uniform(0, 1000, (5, 2))
    dst = src + np.array([7.0, -3.0]) + rng.normal(0, 0.2, (5, 2))

    tform, n_in, rms, model = _fit_transform(src, dst, max_rms=2.0)

    assert model == "translation"
    assert tform.rotation == 0.0
    assert np.allclose(tform.translation, [7.0, -3.0], atol=0.5)
