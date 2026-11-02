"""
field_stars / field_transform: where does a reference position lie in
another frame of the same field?

The cases are the ones measured on real INO340 data: a rich field that
drifts and rotates over a night, a sparse field of five stars that rotates,
and two unrelated fields, which must be refused rather than matched.
"""
import numpy as np
import pytest

from core.align_utils import field_stars, field_transform, _vote_offset

FWHM = 12.0
SIGMA = FWHM / 2.3548


def _rotate(pts, deg, centre, shift):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (pts - centre) @ R.T + centre + np.asarray(shift)


def _image(pts, amps, shape=(1024, 1024), seed=0):
    rng = np.random.default_rng(seed)
    img = 100.0 + rng.normal(0.0, 5.0, shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for (x, y), a in zip(pts, amps):
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            sl = (slice(max(0, int(y) - 40), int(y) + 41), slice(max(0, int(x) - 40), int(x) + 41))
            img[sl] += a * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * SIGMA ** 2))
    return img


def test_field_stars_finds_the_stars_at_full_resolution():
    rng = np.random.default_rng(1)
    truth = rng.uniform(80, 940, (20, 2))
    found = field_stars(_image(truth, rng.uniform(300, 3000, 20)), FWHM)
    assert len(found) >= 18
    d = np.min(np.hypot(*(found[:, None, :] - truth[None, :, :]).transpose(2, 0, 1)), axis=1)
    assert np.median(d) < 0.5, d


def test_rich_field_that_drifts_and_rotates():
    rng = np.random.default_rng(2)
    ref = rng.uniform(0, 4096, (250, 2))
    frame = _rotate(ref, 1.5, np.array([2048.0, 2048.0]), (240.0, -130.0))
    frame += rng.normal(0, 0.3, frame.shape)          # centroid noise
    tform, info = field_transform(frame, ref)
    target_ref = np.array([[2329.0, 1718.0]])
    truth = _rotate(target_ref, 1.5, np.array([2048.0, 2048.0]), (240.0, -130.0))
    pred = tform.inverse(target_ref)
    assert np.hypot(*(pred - truth)[0]) < 0.2, (pred, truth, info)
    assert info["model"] == "similarity"
    assert abs(info["rotation_deg"] + 1.5) < 0.02          # frame -> reference undoes +1.5


def test_sparse_field_of_five_stars_that_rotates():
    # SA 92 2x2 geometry: five stars, the later frames rotated 0.5 deg and
    # shifted (-13, +62); a translation alone leaves ~7 px residuals there
    ref = np.array([[1287.0, 127.0], [1192.0, 1245.0], [914.0, 1549.0],
                    [503.0, 1905.0], [421.0, 716.0]])
    frame = _rotate(ref, -0.5, np.array([1024.0, 1024.0]), (13.0, -62.0))
    tform, info = field_transform(frame, ref)
    pred = tform.inverse(ref)
    assert np.max(np.hypot(*(pred - frame).T)) < 0.1, info
    assert info["model"] == "similarity"


def test_unrelated_fields_are_refused():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        field_transform(rng.uniform(0, 2048, (8, 2)), rng.uniform(0, 2048, (8, 2)))


def test_vote_does_not_depend_on_where_the_shift_falls():
    # five true pairs, shifted so a fixed 8 px histogram would cut them
    # across a bin edge; the bin-free vote still finds them
    ref = np.array([[100.0, 100.0], [400.0, 900.0], [800.0, 300.0], [1500.0, 1200.0], [1900.0, 50.0]])
    jitter = np.array([[-1.5, 1.0], [1.5, -1.2], [-1.2, -1.4], [1.4, 1.3], [0.2, 0.1]])
    shift = np.array([-8.0, 64.0])                       # sits exactly on a bin edge
    frame = ref - shift + jitter
    off = _vote_offset(frame, ref)
    assert off is not None and np.hypot(*(off - shift)) < 2.0


def test_elongated_frame_is_judged_on_the_converged_fit():
    """PSR J2215 High_30 (elongated PSF, FWHM ~19 px) against High_1: real field_stars
    positions. The translation vote's first refit, still missing the 0.26 deg rotation,
    leaves 2.78 px rms; the third refit converges to 0.71 px over 154 stars. The 2 px
    limit used to be applied to every refit, so the frame fell back to "last position"
    (~7.5 px off after the centroid instead of ~1.7 px)."""
    import os
    d = np.load(os.path.join(os.path.dirname(__file__), "data", "psr_high1_high30_field_stars.npz"))
    tform, info = field_transform(d["frame"], d["ref"])
    assert info["rms"] < 1.0 and info["n_inliers"] > 100, info
    pred = tform.inverse(d["star_a_ref"][None, :])[0]
    # exported Table 2 position of star A in High_30 (2-D Gaussian fit): 3.1 px, the
    # elongated-PSF offset of the binned field-star centroids
    assert np.hypot(*(pred - d["star_a_frame"])) < 4.0
