"""
Regression test for core.photometry_core's aperture-photometry noise formula,
checked against the standard CCD/CMOS aperture-photometry SNR equation AND,
independently, against the textbook pure-shot-noise limit (SNR = sqrt(N_e)),
which needs no external reference at all.

References:
  - Vik Dhillon (Univ. of Sheffield), PHY217 "The CCD Equation":
    https://vikdhillon.staff.shef.ac.uk/teaching/phy217/detectors/phy217_det_ccdeqn.html
  - Newberry (1991); Mighell, K.J. (1999), PASP 111, 649, "Algorithms for CCD
    Stellar Photometry"; as implemented by AstroImageJ's "revised CCD equation"
    (Collins et al. 2017, AJ 153, 77 -- https://arxiv.org/abs/1601.02622).

The standard equation, done entirely in ELECTRON units (all sources agree on
this form):

    signal_e = net_counts_ADU * gain
    noise_sq_e = (net_counts_ADU * gain)            # source shot noise
               + (bkg_mean_ADU * gain * n_ap)        # sky shot noise
               + (dark_noise * n_ap)                 # dark current (already e-)
               + (read_noise**2 * n_ap)              # read noise (already e-)
    SNR = signal_e / sqrt(noise_sq_e)

perform_aperture_photometry computes this in an ADU^2-variance-domain form
that must give the IDENTICAL ratio:

    noise_sq = source_flux/gain + (read_noise**2 + dark_noise)*n_ap/gain**2
    SNR = net_counts / sqrt(noise_sq)      # net_counts, NOT net_counts/gain

Three bugs were found and fixed here, in order of discovery:
  1. Background double-counted: source_flux (sky-inclusive) already carries
     the sky's Poisson contribution; a separate `+ bkg_mean*n_ap` term
     counted it twice.
  2. Wrong power of gain on read/dark noise: they're already in electrons
     (confirmed against the FLI Kepler KL4040 datasheet, 3.7 e- read noise
     in High mode, and this project's own std(Bias1-Bias2)/sqrt(2) measurement),
     so they need gain**2, not gain**1, to reach the ADU^2 domain.
  3. Spurious extra gain division on the signal itself: `net_signal =
     net_counts/gain` introduced one more division by gain than the (already
     gain**2-corrected) noise term accounted for, making the reported SNR
     exactly `gain` times too small in every regime -- this was caught only
     by cross-checking the ABSOLUTE SNR value against an independently
     derived reference number, not merely the internal structure of the fix.

This file verifies the fix two independent ways so bug #3's failure mode
(self-consistent but absolutely wrong by a constant factor) cannot recur
undetected:
  (a) the textbook pure-shot-noise limit: with zero sky, zero read noise,
      zero dark current, SNR must equal sqrt(net_counts * gain) exactly --
      this requires no external formula, just the definition of Poisson
      shot noise.
  (b) the full standard equation above, computed independently in the
      electron domain (not by reusing perform_aperture_photometry's own
      ADU^2-domain algebra), for a realistic sky+read+dark scenario.
"""
import numpy as np
import pytest

from core.photometry_core import perform_aperture_photometry


def _make_synthetic_star(ny=201, nx=201, cy=100.0, cx=100.0,
                          bkg_level=500.0, amplitude=2000.0, sigma=3.0):
    """Deterministic flat-sky + Gaussian-star image. No noise added, so aperture
    sums are exact and the test is 100% reproducible."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    star = amplitude * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    return (bkg_level + star).astype('float64')


def test_pure_shot_noise_limit_matches_textbook_sqrt_n():
    """Zero sky, zero read noise, zero dark current: SNR must equal
    sqrt(net_counts * gain) -- the textbook definition of shot-noise-limited
    SNR for a Poisson process. No external reference needed for this check."""
    gain = 16.5
    aperture_radius = 10.0
    image = _make_synthetic_star(bkg_level=0.0, amplitude=5000.0, sigma=3.0)
    center = (100.0, 100.0)

    res = perform_aperture_photometry(
        image, center, aperture_radius,
        inner_coef=2.0, outer_coef=3.0,
        read_noise=0.0, dark_noise=0.0, gain=gain,
        zeropoint=0.0, exptime=1.0,
    )

    expected_snr = np.sqrt(res['flux'] * gain)
    print(f"\n[pure shot-noise limit] snr={res['snr']:.4f}  sqrt(N_e)={expected_snr:.4f}")
    assert np.isclose(res['snr'], expected_snr, rtol=1e-6)


@pytest.mark.parametrize("amplitude,bkg_level,label", [
    (20000.0, 500.0, "bright star over normal sky"),
    (150.0, 500.0, "faint, sky-dominated star"),
    (80.0, 0.05, "near-zero background, faint star"),
])
def test_perform_aperture_photometry_matches_standard_ccd_equation(amplitude, bkg_level, label):
    gain = 16.5
    read_noise = 3.18
    dark_noise = 4.15
    aperture_radius = 10.0

    image = _make_synthetic_star(amplitude=amplitude, bkg_level=bkg_level, sigma=3.0)
    center = (100.0, 100.0)

    res = perform_aperture_photometry(
        image, center, aperture_radius,
        inner_coef=2.0, outer_coef=3.0,
        read_noise=read_noise, dark_noise=dark_noise, gain=gain,
        zeropoint=0.0, exptime=1.0,
    )

    net_counts = res['flux']
    bkg_mean = res['bkg_mean']
    n_ap = res['aperture_area']

    # Independent reference computation, entirely in the ELECTRON domain --
    # deliberately NOT reusing perform_aperture_photometry's own ADU^2-domain
    # algebra, so a shared mistake in that algebra can't hide from this test.
    signal_e = net_counts * gain
    noise_sq_e = (net_counts * gain) + (bkg_mean * gain * n_ap) + (dark_noise * n_ap) + (read_noise ** 2 * n_ap)
    expected_snr = signal_e / np.sqrt(noise_sq_e)
    expected_mag_err = (2.5 / np.log(10)) / expected_snr  # exact 1.0857/SNR

    print(f"\n[{label}] snr={res['snr']:.4f} (expected {expected_snr:.4f}), "
          f"mag_err={res['mag_err']:.5f} (expected {expected_mag_err:.5f})")

    assert np.isclose(res['snr'], expected_snr, rtol=1e-6), (
        f"perform_aperture_photometry SNR does not match the standard CCD "
        f"equation (electron-domain, independently computed) for case '{label}'."
    )
    assert np.isclose(res['mag_err'], expected_mag_err, rtol=1e-6)
