"""
A detection has to look like a star before it is believed.

Nothing used to check. `twoD_Gaussian_fit` returned wherever LevMar stopped and
the callers accepted it on `fit is not None` alone, so on pure Gaussian noise
with no source, `find_star_in_frame` on a 600 px stamp returned:

    found: (-340.40, 628.50)   amplitude 22.5, x_std 125.2, y_std 8.4

-- a centre off the array, with a 125 px width, reported as a detection. The
bulk worker then skipped alignment for that frame (`aligned = data`), fed the
position forward as the next frame's search seed, and measured photometry
there.
"""
import numpy as np
import pytest

from core.align_utils import find_star_in_frame, find_star_adaptive

SKY, SKY_RMS = 100.0, 3.0
TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def frame(size=200, seed=0):
    return np.random.default_rng(seed).normal(SKY, SKY_RMS, (size, size))


def add_star(img, x, y, peak, fwhm):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    sig = fwhm / TO_FWHM
    return img + peak * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2)))


def test_blank_noise_is_not_a_star():
    """The exact case that used to return a centre outside the image."""
    found, _, _ = find_star_in_frame(frame(600), (300, 300), stamp_radius=300,
                                     fwhm_guess=15)
    assert found is None


def test_smooth_gradient_is_not_a_star():
    yy, xx = np.mgrid[0:200, 0:200]
    img = SKY + 0.5 * xx + np.random.default_rng(1).normal(0, SKY_RMS, (200, 200))
    assert find_star_in_frame(img, (100, 100), stamp_radius=60,
                              fwhm_guess=15)[0] is None


def test_single_hot_pixel_is_not_a_star():
    """A one-pixel spike is far below the stellar width floor."""
    img = frame(seed=2)
    img[100, 100] += 5000.0
    assert find_star_in_frame(img, (100, 100), stamp_radius=60,
                              fwhm_guess=15)[0] is None


@pytest.mark.parametrize("true_xy, fwhm", [((300.0, 300.0), 15.0),
                                           ((312.4, 288.7), 15.0),
                                           ((300.0, 300.0), 8.0)])
def test_real_star_is_found_to_subpixel(true_xy, fwhm):
    img = add_star(frame(600, seed=3), true_xy[0], true_xy[1], 4000.0, fwhm)
    found, params, _ = find_star_in_frame(img, (300, 300), stamp_radius=60,
                                          fwhm_guess=15)
    assert found is not None
    assert np.hypot(found[0] - true_xy[0], found[1] - true_xy[1]) < 0.1


@pytest.mark.parametrize("fwhm", [8.0, 15.0])
def test_measured_width_is_not_inflated_by_smoothing(fwhm):
    """
    The fit runs on the unsmoothed stamp.

    It used to run on a copy smoothed by sigma = fwhm_guess/3, which adds its
    own width in quadrature -- sigma_fit^2 = sigma_true^2 + sigma_smooth^2 --
    and returned ~19 px for a 15 px PSF. The centroid was unaffected; the width
    was not, and the width is what the validity checks rely on.
    """
    img = add_star(frame(600, seed=4), 300.0, 300.0, 4000.0, fwhm)
    _, params, _ = find_star_in_frame(img, (300, 300), stamp_radius=60,
                                      fwhm_guess=15)
    measured = TO_FWHM * 0.5 * (abs(params['x_std']) + abs(params['y_std']))
    assert abs(measured - fwhm) < 0.3


def test_a_broad_source_is_not_rejected_for_a_low_fitted_amplitude():
    """
    A peak-brightness bar throws away strong, real sources.

    The source near (3138, 2242) on the GRB 22-Oct i frame is a 37.5-sigma
    matched-filter detection, yet its FITTED amplitude is only 1.9-2.4x the
    stamp's background rms at every seed, stamp radius and FWHM guess: the
    17.5 px PSF still has flux at the stamp edge, where estimate_background
    reads the sky, so part of the star is counted as sky (raw peak 150 ADU,
    fitted 82). The 3-sigma bar that used to sit here rejected it, and that is
    what drove the operator to fwhm=5, threshold=0.1 to get anything at all.
    """
    img = add_star(frame(seed=5), 100.0, 100.0, 2 * SKY_RMS, 15.0)
    found, params, _ = find_star_in_frame(img, (100, 100), stamp_radius=40,
                                          fwhm_guess=15)
    assert found is not None
    assert np.hypot(found[0] - 100.0, found[1] - 100.0) < 3.0


def test_a_fit_that_wanders_far_from_the_seed_is_rejected():
    """
    The discriminator, since brightness cannot be one. Over 60 pure-noise
    stamps the fits land a median 43 px from the seed and only 1 in 60 within
    10 px, while the real source lands 6.6 px from the click -- the same
    6.6 px whether the FWHM guess is 5, 14 or 17.5. A star placed well outside
    the allowed offset must not be returned as if it were at the seed.
    """
    img = add_star(frame(seed=8), 160.0, 160.0, 4000.0, 14.0)   # far corner
    assert find_star_in_frame(img, (100, 100), stamp_radius=40,
                              fwhm_guess=14)[0] is None


def test_expand_steps_is_obeyed():
    """
    `expand_steps=(0,)` means ONE stamp.

    An unconditional extra rung at stamp_radius + 100 used to run after the
    loop, so the bulk worker's "strict, no jumping" refinement searched a
    261x261 stamp anyway: on the GRB 22-Oct i frame a candidate 7 px from the
    click came back 104 px away, on a different star.
    """
    img = add_star(frame(600, seed=9), 300.0, 300.0, 4000.0, 14.0)
    img = add_star(img, 400.0, 300.0, 40000.0, 14.0)      # much brighter, 100 px away
    found, _, _ = find_star_in_frame(img, (300, 300), stamp_radius=25,
                                     fwhm_guess=14, expand_steps=(0,))
    assert found is not None
    assert abs(found[0] - 300.0) < 2.0, "refinement jumped to the brighter star"


def test_adaptive_search_follows_a_drifting_star():
    """Tracking uses the previous frame's position as the next frame's seed."""
    seed_xy = (100.0, 100.0)
    prev = None
    for i in range(6):
        x, y = 100.0 + 3.0 * i, 100.0 - 2.0 * i
        img = add_star(frame(seed=10 + i), x, y, 3000.0, 14.0)
        found, _, _ = find_star_adaptive(img, seed_xy, previous_detected_xy=prev,
                                         stamp_radius=40, fwhm_guess=14)
        assert found is not None, f"lost the star on frame {i}"
        assert np.hypot(found[0] - x, found[1] - y) < 0.5
        prev = found


def test_adaptive_search_reports_nothing_when_the_star_is_gone():
    """A frame with no star must not inherit the previous frame's position.

    Pure noise is not rejected every single time -- a Gaussian fit finds a
    plausible bump near the seed in about 4 stamps in 60 -- so this uses a
    handful of seeds and requires the great majority to come back empty.
    """
    misses = sum(
        find_star_adaptive(frame(seed=20 + k), (100.0, 100.0),
                           previous_detected_xy=(100.0, 100.0),
                           stamp_radius=40, fwhm_guess=14)[0] is None
        for k in range(12))
    assert misses >= 10, f"only {misses}/12 blank frames reported no star"
