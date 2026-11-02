"""
estimate_frame_fwhm on a SPARSE frame: few faint stars, many cosmic rays.

Found on GRB 251013C, g band, KL4040 Low-gain frame 1 (sky 8 ADU, sigma 2.1,
3 usable stars, true FWHM ~20 px by a Moffat fit). Pressing "Measure" with
Fit FWHM = 12 returned 12.19 px; pressing it again (guess 12.19) returned
20.04 px. Each call took 18-28 s on the GUI thread, because the sample box grew
to 2048 px and DAOStarFinder convolved it directly with 24-48 px kernels.

The synthetic frame below has the same ingredients: a low integer-quantised
sky, a handful of faint Moffat stars spread over the frame (so the central
sample box finds too few and has to grow), and compact cosmic-ray hits and hot
pixels. sample_box is scaled down (256 px on 1024 px) to keep the test fast.
"""
import numpy as np
import pytest
from photutils.detection import DAOStarFinder

import core.photometry_core as pc

# A rung that finds nothing is expected here (that is part of what is tested).
pytestmark = pytest.mark.filterwarnings("ignore::photutils.utils.exceptions.NoDetectionsWarning")

SKY, READ_NOISE, GAIN = 8.0, 1.9, 18.7      # ADU, ADU, e-/ADU: KL4040 Low gain
TRUE_FWHM = 12.0
GUESSES = (6.0, 8.0, 12.0, 12.19, 20.0, 25.0)


def moffat(shape, x0, y0, fwhm, beta, peak):
    gamma = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    yy, xx = np.indices(shape, dtype=float)
    return peak * (1.0 + ((xx - x0) ** 2 + (yy - y0) ** 2) / gamma ** 2) ** (-beta)


def sparse_frame(n=1024, fwhm=12.0, beta=3.5, n_stars=6, peaks=(15.0, 100.0),
                 n_cosmics=20, seed=1):
    rng = np.random.default_rng(seed)
    model = np.full((n, n), SKY)
    half = int(6 * fwhm)
    for (x, y), p in zip(rng.uniform(100, n - 100, size=(n_stars, 2)),
                         np.geomspace(peaks[0], peaks[1], n_stars)):
        xi, yi = int(x), int(y)
        y0, y1, x0, x1 = max(0, yi - half), min(n, yi + half), max(0, xi - half), min(n, xi + half)
        model[y0:y1, x0:x1] += moffat((y1 - y0, x1 - x0), x - x0, y - y0, fwhm, beta, p)
    img = rng.poisson(model * GAIN) / GAIN + rng.normal(0.0, READ_NOISE, model.shape)
    for k in range(n_cosmics):                # hot pixels, short tracks, 2x2 hits
        x, y = rng.integers(20, n - 20, size=2)
        amp = rng.uniform(40.0, 160.0)
        if k % 3 == 0:
            img[y, x] += 1.5 * amp
        elif k % 3 == 1:
            dx, dy = rng.choice([-1, 0, 1], size=2)
            dy = dy if (dx or dy) else 1
            for s in range(rng.integers(2, 5)):
                img[y + s * dy, x + s * dx] += amp * rng.uniform(0.4, 1.0)
        else:
            img[y:y + 2, x:x + 2] += 0.5 * amp
            img[y + 2, x] += 0.15 * amp
    return np.round(img)                      # calibrated frames are stored as integers


def test_no_detection_convolves_much_more_than_the_first_box(monkeypatch):
    """
    DAOStarFinder's cost is pixels x kernel area. Before, a grown box was
    searched at full resolution with kernels up to 4 x guess: here the whole
    1024^2 frame with a 48 px kernel, i.e. 16x the pixels of the first box.
    """
    calls = []

    class Recording(DAOStarFinder):
        def __call__(self, data, mask=None):
            calls.append((self.fwhm, data.size))
            return super().__call__(data, mask=mask)

    monkeypatch.setattr(pc, "DAOStarFinder", Recording)
    img = sparse_frame()
    for guess in (6.0, 12.0, 25.0):
        pc.estimate_frame_fwhm(img, fwhm_guess=guess, sample_box=256)

    assert calls, "DAOStarFinder was never called"
    assert max(px for _, px in calls) <= 4 * 256 ** 2
    assert max(fw for fw, _ in calls) < 32.0


@pytest.mark.parametrize("seed", [3, 5])
def test_sparse_frame_answer_does_not_depend_on_the_guess(seed):
    """
    Cosmic rays must not outvote the few stars. Before: seed 3 returned 7.78 px
    at guess 25 (true 12): the 50 px rung found a cosmic ray and no star, and
    the median of two disagreeing rungs is their midpoint. Seed 5, with the
    grown box searched on a binned copy but no artefact handling, returned
    5.71 px at guess 6: cosmic rays were half of a rung's sample.
    """
    img = sparse_frame(fwhm=TRUE_FWHM, seed=seed)
    got = [pc.estimate_frame_fwhm(img, fwhm_guess=g, sample_box=256) for g in GUESSES]
    assert None not in got, got
    got = np.array(got)
    assert np.all(np.abs(got - TRUE_FWHM) <= 0.05 * TRUE_FWHM), got
    assert got.max() - got.min() <= 0.02 * np.median(got), got
