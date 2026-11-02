"""
Tests for core.ext_coeff -- now the single implementation used by the GUI
(the inline duplicate in multi_photometry_controller was removed).
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from core import ext_coeff


def test_parse_skycoord_decimal_degrees_not_misread_as_hours():
    sc = ext_coeff._parse_skycoord("45.3", "20.0")
    assert sc is not None
    # Old behavior parsed 45.3 as HOURS (-> 679.5 deg wrapped); must be degrees.
    assert abs(sc.ra.deg - 45.3) < 1e-6
    assert abs(sc.dec.deg - 20.0) < 1e-6


def test_parse_skycoord_sexagesimal_hourangle():
    sc = ext_coeff._parse_skycoord("22:15:30", "+51:35:10")
    assert sc is not None
    assert abs(sc.ra.deg - (22 + 15 / 60 + 30 / 3600) * 15.0) < 1e-6


def test_ut_is_taken_from_date_not_dateobs():
    """
    On KL4040 headers DATE is UT and DATE-OBS is Iran LOCAL time.

    This test used to assert the opposite -- and its own example values are
    3 h 30 m apart, i.e. they were copied from a real header and mislabelled.
    Reading DATE-OBS as UT put every airmass 3 h 30 m out in hour angle. On
    the 12 real GRB 251013C g-band frames that placed the target 5-7 degrees
    BELOW the horizon, so every frame was skipped and the extinction fit
    could not run at all.
    """
    hdr = fits.Header()
    hdr['DATE'] = '2025-10-08T20:07:56.923000'          # UT
    hdr['DATE-OBS'] = '2025-10-08T23:37:56.185 NOGPS'   # Iran local (UT+3:30)
    dt = ext_coeff.get_datetime_from_header(hdr)
    assert dt is not None
    assert dt.startswith('2025-10-08T20:07:56'), (
        "the two clocks differ by a civil timezone offset, so DATE is the UT one")


def test_dateobs_wins_when_date_is_a_file_creation_stamp():
    """An arbitrary difference means DATE really is when the file was written."""
    hdr = fits.Header()
    hdr['DATE'] = '2025-10-09T11:42:13'                 # written next morning
    hdr['DATE-OBS'] = '2025-10-08T23:37:56'
    assert ext_coeff.get_datetime_from_header(hdr).startswith('2025-10-08T23:37:56')


def test_single_time_key_is_used_as_is():
    hdr = fits.Header()
    hdr['DATE-OBS'] = '2025-10-08T23:37:56'
    assert ext_coeff.get_datetime_from_header(hdr).startswith('2025-10-08T23:37:56')
    hdr2 = fits.Header()
    hdr2['DATE'] = '2025-10-08T20:07:56'
    assert ext_coeff.get_datetime_from_header(hdr2).startswith('2025-10-08T20:07:56')


def test_airmass_uses_the_exposure_midpoint(tmp_path):
    """
    The header time is the START of the exposure; the airmass that belongs
    with the frame's flux is the one at mid-exposure.
    """
    from astropy.io import fits as _fits

    hdr = _fits.Header()
    hdr['DATE'] = '2025-10-13T21:04:17'
    hdr['EXPTIME'] = (5813953, 'Duration of exposure in 10us periods')  # 58.14 s
    path = str(tmp_path / "f.fits")
    _fits.PrimaryHDU(np.zeros((4, 4), dtype='float32'), header=hdr).writeto(path)

    mid, _ = ext_coeff.compute_airmass_for_files(
        [path], [0], 345.8146, -0.2056, 33.4, 51.19, elevation_m=3600.0,
        use_exposure_midpoint=True)
    start, _ = ext_coeff.compute_airmass_for_files(
        [path], [0], 345.8146, -0.2056, 33.4, 51.19, elevation_m=3600.0,
        use_exposure_midpoint=False)

    assert mid and start
    assert mid[0]['exptime_s'] == pytest.approx(58.13953, abs=1e-4)
    shift_s = (Time(mid[0]['time_iso']) - Time(start[0]['time_iso'])).to_value('sec')
    assert shift_s == pytest.approx(58.13953 / 2.0, abs=0.01)
    assert mid[0]['airmass'] != start[0]['airmass']


def test_fit_recovers_known_extinction_slope():
    rng = np.random.default_rng(42)
    true_k, true_zp = 0.25, 12.0
    airmass = np.linspace(1.0, 2.5, 30)
    sigma = 0.01
    mags = true_zp + true_k * airmass + rng.normal(0, sigma, airmass.size)
    errs = np.full_like(airmass, sigma)

    fit = ext_coeff.fit_mag_vs_airmass(list(mags), list(errs), list(airmass))
    assert abs(fit['slope'] - true_k) < 5 * fit['slope_err']
    assert abs(fit['slope'] - true_k) < 0.02
    # With absolute errors and cov='unscaled', slope_err should be in the
    # right ballpark (not wildly rescaled).
    assert 0.001 < fit['slope_err'] < 0.05


def test_fit_reports_chi2_nu_and_scaled_error():
    rng = np.random.default_rng(20261017)
    x = np.linspace(1.2, 2.2, 400)
    sigma = np.full_like(x, 0.01)
    for scatter, lo, hi in ((0.01, 0.8, 1.2), (0.03, 7.5, 10.5)):     # 1x and 3x the error bars
        y = 20.0 + 0.2 * x + rng.normal(0.0, scatter, x.size)
        fit = ext_coeff.fit_mag_vs_airmass(list(y), list(sigma), list(x))
        assert lo < fit['chi2_nu'] < hi, (scatter, fit['chi2_nu'])
        assert abs(fit['slope_err_scaled'] / fit['slope_err'] - np.sqrt(fit['chi2_nu'])) < 1e-9


def test_unweighted_fit_has_no_chi2():
    fit = ext_coeff.fit_mag_vs_airmass([20.1, 20.3, 20.5], None, [1.0, 2.0, 3.0])
    assert fit['chi2_nu'] is None and fit['slope_err_scaled'] is None


def test_plot_dialog_shows_chi2_and_the_scaled_error():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from views.components.extinction_plot_dialog import ExtinctionPlotDialog
    x = np.array([1.6, 1.8, 2.0]); y = np.array([-6.3, -6.28, -6.25]); e = np.full(3, 0.004)
    texts = lambda d: [w.text() for w in d.findChildren(QtWidgets.QLabel)]
    d = ExtinctionPlotDialog(None, x, y, e, 0.16482, -6.5968, 0.00863,
                             chi2_nu=80.74, k_err_scaled=0.07755)
    assert "80.74" in texts(d) and "\u00b1 0.07755" in texts(d)
    d2 = ExtinctionPlotDialog(None, x, y, e, 0.16482, -6.5968, 0.00863,
                              chi2_nu=0.8, k_err_scaled=0.0077)
    assert "0.80" in texts(d2) and not any("0.00770" in t for t in texts(d2))
