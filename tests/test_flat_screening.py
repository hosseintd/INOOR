"""
Screening a flat set: which frames must not go into the master.

The four tests are deliberately independent, and each one here is exercised in
isolation so a regression names the failure mode it broke.

The most important case is the LOCALISED one. A first attempt judged the
consensus with a robust (MAD) scatter alone, which is blind to a defect
covering less than half the frame -- a dead quadrant is exactly 25%, and the
MAD of a field that is three-quarters zero stays near zero. A synthetic
quadrant glitch sailed through. The block-median test exists because of that.
"""
import numpy as np
import pytest

from core import flat_screening as fsc


def _flat(level=2000.0, shape=(512, 512), seed=0):
    """A plausible flat: smooth vignetting, photon noise, no stars."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    r2 = ((y - ny / 2) ** 2 + (x - nx / 2) ** 2) / (nx / 2) ** 2
    vignette = 1.0 - 0.25 * r2
    ideal = level * vignette
    return (ideal + rng.normal(0, np.sqrt(np.maximum(ideal, 1)))).astype('float32')


def _screen(frames, **kw):
    store = {f"f{i}.fits": a for i, a in enumerate(frames)}
    return fsc.screen_flat_frames(list(store), lambda p: store[p], **kw)


def _verdicts(res):
    return [r['ok'] for r in res['frames']]


def test_a_clean_set_passes():
    res = _screen([_flat(2000 + 100 * i, seed=i) for i in range(6)])
    assert all(_verdicts(res)), [r['reasons'] for r in res['frames']]


def test_a_wide_but_honest_level_range_still_passes():
    """Twilight flats drift a lot in level. That alone is not a fault."""
    res = _screen([_flat(lvl, seed=i)
                   for i, lvl in enumerate([600, 1000, 1600, 2200, 2800, 3300])])
    assert all(_verdicts(res)), [r['reasons'] for r in res['frames']]


def test_a_frame_above_the_linearity_limit_is_rejected():
    # The level argument is the peak; vignetting pulls the median down, so aim
    # high enough that the MEDIAN clears the limit -- that is what is tested.
    hot = _flat(4400, seed=9)
    assert np.median(hot) > fsc.LINEARITY_LIMIT_ADU, "test frame is not hot enough"
    res = _screen([_flat(2000, seed=i) for i in range(5)] + [hot])
    assert _verdicts(res)[-1] is False
    assert any("linearity" in w for w in res['frames'][-1]['reasons'])


def test_a_frame_with_no_signal_is_rejected():
    frames = [_flat(2000, seed=i) for i in range(5)] + [_flat(100, seed=9)]
    res = _screen(frames)
    assert _verdicts(res)[-1] is False
    assert any("too low" in w for w in res['frames'][-1]['reasons'])


def test_saturated_pixels_are_rejected():
    bad = _flat(3000, seed=9)
    bad[:60, :60] = 4200.0                       # ~1.4% pinned at the ceiling
    res = _screen([_flat(2000, seed=i) for i in range(5)] + [bad])
    assert _verdicts(res)[-1] is False
    assert any("ceiling" in w for w in res['frames'][-1]['reasons'])


def test_a_star_field_is_rejected():
    rng = np.random.default_rng(3)
    bad = _flat(2000, seed=9)
    ys, xs = rng.integers(10, 500, 3000), rng.integers(10, 500, 3000)
    bad[ys, xs] += 900.0
    res = _screen([_flat(2000, seed=i) for i in range(5)] + [bad])
    assert _verdicts(res)[-1] is False
    assert any("spikes" in w for w in res['frames'][-1]['reasons'])


def test_a_quadrant_glitch_is_rejected():
    """The case a global scatter statistic cannot see."""
    bad = _flat(2000, seed=9)
    bad[:256, :256] *= 0.88                      # one quadrant, 12% low
    res = _screen([_flat(2000, seed=i) for i in range(5)] + [bad])
    assert _verdicts(res)[-1] is False, (
        "a defect over a quarter of the frame must be caught; "
        "a MAD-based scatter alone cannot see it")
    assert any("part of the frame" in w for w in res['frames'][-1]['reasons'])


def test_junk_frames_do_not_become_the_reference():
    """
    A folder with a few good flats and a pile of junk must flag the junk, not
    the flats. The consensus is therefore built only from frames that already
    passed the absolute tests.
    """
    good = [_flat(2000, seed=i) for i in range(3)]
    junk = [_flat(80, seed=100 + i) for i in range(7)]     # all below min level
    res = _screen(good + junk)
    verdicts = _verdicts(res)
    assert verdicts[:3] == [True, True, True], (
        f"the good flats were flagged: {[r['reasons'] for r in res['frames'][:3]]}")
    assert not any(verdicts[3:])


def test_every_measurement_is_reported_not_just_the_verdict():
    res = _screen([_flat(2000, seed=i) for i in range(4)])
    r = res['frames'][0]
    for key in ('median', 'sat_fraction', 'spikes', 'consensus_sigma',
                'block_deviation'):
        assert key in r and np.isfinite(r[key]), key
    assert res['thresholds']['linearity_limit'] == fsc.LINEARITY_LIMIT_ADU


def test_too_few_frames_skips_the_relative_tests_and_says_so():
    res = _screen([_flat(2000, seed=0), _flat(2100, seed=1)])
    assert "at least 3" in res['summary']
    assert all(_verdicts(res))


def test_an_unreadable_frame_is_reported_not_swallowed():
    def loader(p):
        if p == "f1.fits":
            raise OSError("truncated file")
        return _flat(2000)
    res = fsc.screen_flat_frames(["f0.fits", "f1.fits", "f2.fits"], loader)
    assert res['frames'][1]['ok'] is False
    assert "cannot be read" in res['frames'][1]['reasons'][0]


def test_empty_input_is_harmless():
    res = fsc.screen_flat_frames([], lambda p: None)
    assert res['frames'] == []
