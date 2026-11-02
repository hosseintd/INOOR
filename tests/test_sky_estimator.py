"""
The sky under an aperture is a 3-sigma-clipped mean of the annulus pixels.

A plain mean lets a neighbour's core or a cosmic ray in the annulus raise the
sky under the whole aperture: on PSR J2215 frame 86 a faint star whose 5x
brighter neighbour sat in its 2r-3r annulus read 0.80 mag too faint.
"""
import numpy as np
import pytest

from core import photometry_core as pc

SKY, NOISE = 40.0, 8.0


def _field(neighbour=True, seed=20261012):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:301, 0:301]
    img = SKY + rng.normal(0.0, NOISE, (301, 301))
    sig = 12.0 / 2.3548
    img += 300.0 * np.exp(-((xx - 150) ** 2 + (yy - 150) ** 2) / (2 * sig ** 2))
    if neighbour:   # 5x brighter star 30 px away: inside the 24-36 px annulus
        img += 1500.0 * np.exp(-((xx - 180) ** 2 + (yy - 150) ** 2) / (2 * sig ** 2))
    return img


def test_a_neighbour_in_the_annulus_does_not_raise_the_sky():
    img = _field(neighbour=True)
    sky, n = pc.annulus_sky(img, (150.0, 150.0), 24.0, 36.0)
    # The neighbour's wing pixels below 3 sigma survive the clip by design, so
    # ~1 ADU of it remains -- against ~+90 ADU for the plain mean.
    assert abs(sky - SKY) < 1.5, f"clipped sky {sky:.2f}, true {SKY}"
    # The plain mean of the same pixels is what INOOR used to subtract.
    yy, xx = np.indices(img.shape)
    r = np.hypot(xx - 150.0, yy - 150.0)
    plain = img[(r >= 24) & (r < 36)].mean()
    assert plain - SKY > 5.0, "the test field must actually contaminate a plain mean"

    clean = pc.perform_aperture_photometry(_field(neighbour=False), (150.0, 150.0), 12.0,
                                           gain=1.0, read_noise=3.8, exptime=1.0)
    dirty = pc.perform_aperture_photometry(img, (150.0, 150.0), 12.0,
                                           gain=1.0, read_noise=3.8, exptime=1.0)
    assert dirty["flux"] == pytest.approx(clean["flux"], rel=0.03)


def test_snr_curve_and_measurement_share_the_sky_estimator():
    img = _field(neighbour=True)
    radii = np.array([8.0, 12.0, 16.0])
    _, snrs, _, _ = pc.compute_snr_vs_radius(img, (150.0, 150.0), radii=radii,
                                             gain=1.0, read_noise=3.8, dark_noise=4.15)
    for r, s in zip(radii, snrs):
        m = pc.perform_aperture_photometry(img, (150.0, 150.0), r, gain=1.0,
                                           read_noise=3.8, dark_noise=4.15, exptime=1.0)
        assert s == pytest.approx(m["snr"], rel=1e-9)


def test_an_annulus_off_the_edge_averages_the_pixels_it_has():
    img = SKY + np.random.default_rng(1).normal(0.0, NOISE, (200, 200))
    sky, n = pc.annulus_sky(img, (5.0, 100.0), 20.0, 30.0)
    assert 0 < n < np.pi * (30 ** 2 - 20 ** 2)
    assert abs(sky - SKY) < 1.5


def test_no_finite_pixel_gives_nan_not_zero():
    assert pc.clipped_sky_mean([]) == (pytest.approx(float("nan"), nan_ok=True), 0)
    assert pc.clipped_sky_mean([np.nan, np.nan])[1] == 0
