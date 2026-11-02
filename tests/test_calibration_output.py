"""Regression tests for calibrated-frame arithmetic and FITS output.

Two historical bugs are pinned here:

  1. The output writer clipped every calibrated frame to 0..4095 uint16. After
     dark subtraction and flat division pixels legitimately fall below 0 (the
     negative wing of the sky noise) and above 4095 (a bright star in a
     vignetted corner, where norm_flat < 1 scales counts UP). Clipping invented
     an artificial dynamic range. Output must now preserve the full range while
     staying at 2 bytes/pixel via BSCALE/BZERO.

  2. calibrate() computed  lf - mb - clip(md - mb, min=0)  which equals lf - md
     everywhere EXCEPT where md < mb, and additionally clipped the result at
     zero. Both truncations biased the background upward.
"""
import numpy as np
import pytest
from astropy.io import fits

from core import calibration as calib
from core import masterFrame_creator as mfc


# ---------------------------------------------------------------- pedestal --

def test_master_dark_carries_the_bias():
    lf = np.full((4, 4), 1000.0, dtype='float32')
    mb = np.full((4, 4), 300.0, dtype='float32')
    md = np.full((4, 4), 305.0, dtype='float32')  # bias + 5 ADU dark current
    # Subtracting the dark removes bias AND dark current; the bias must not be
    # subtracted a second time.
    assert calib.subtract_pedestal(lf, md, mb)[0, 0] == pytest.approx(695.0)


def test_bias_only_run_still_subtracts_the_bias():
    lf = np.full((4, 4), 1000.0, dtype='float32')
    mb = np.full((4, 4), 300.0, dtype='float32')
    assert calib.subtract_pedestal(lf, None, mb)[0, 0] == pytest.approx(700.0)


def test_no_masters_leaves_the_frame_alone():
    lf = np.full((4, 4), 1000.0, dtype='float32')
    assert calib.subtract_pedestal(lf, None, None)[0, 0] == pytest.approx(1000.0)


def test_negative_sky_is_not_clipped_to_zero():
    """The old code clamped this to 0, truncating the noise distribution."""
    lf = np.full((4, 4), 250.0, dtype='float32')
    md = np.full((4, 4), 305.0, dtype='float32')
    out = calib.subtract_pedestal(lf, md, np.full((4, 4), 300.0, dtype='float32'))
    assert out[0, 0] == pytest.approx(-55.0)


def test_dark_below_bias_is_not_clipped():
    """Where read noise puts md below mb, the old clip left lf - mb instead of
    lf - md, leaving a systematic positive residual."""
    lf = np.full((4, 4), 1000.0, dtype='float32')
    mb = np.full((4, 4), 300.0, dtype='float32')
    md = np.full((4, 4), 297.0, dtype='float32')  # noise dip below the bias
    assert calib.subtract_pedestal(lf, md, mb)[0, 0] == pytest.approx(703.0)


def test_flat_normalisation_uses_the_median():
    mf = np.full((8, 8), 2000.0, dtype='float32')
    mf[0, 0] = 1e6  # hot pixel must not shift the normalisation
    norm = calib.normalize_flat(mf)
    assert norm[4, 4] == pytest.approx(1.0)


def test_dead_flat_pixels_are_not_amplified():
    """A divisor clamped at eps=1e-6 multiplied the science frame by up to 1e6.
    Subtracting the flat's pedestal pushes genuinely dead pixels negative, which
    turned that latent bug into +/-1e8 spikes. Dead pixels must be left
    uncorrected instead."""
    mf = np.full((16, 16), 1.0, dtype='float32')
    mf[0, 0] = -0.8     # dead pixel, negative after pedestal removal
    mf[1, 1] = 0.0
    mf[2, 2] = 0.05     # nearly dead

    norm = calib.normalize_flat(mf)
    assert norm[0, 0] == 1.0
    assert norm[1, 1] == 1.0
    assert norm[2, 2] == 1.0
    assert norm[8, 8] == pytest.approx(1.0)

    light = np.full((16, 16), 120.0, dtype='float32')
    out = calib.calibrate(light, None, None, mf)
    assert np.abs(out).max() < 1e3, "dead flat pixel amplified the science frame"


def test_real_vignetting_is_not_mistaken_for_dead_pixels():
    """The dead-pixel floor must sit well below genuine vignetting and dust."""
    mf = np.full((16, 16), 1.0, dtype='float32')
    mf[0, :] = 0.70     # vignetted corner
    mf[1, :] = 0.45     # dust donut
    norm = calib.normalize_flat(mf)
    assert norm[0, 0] == pytest.approx(0.70, rel=1e-3)
    assert norm[1, 0] == pytest.approx(0.45, rel=1e-3)


# ------------------------------------------------------------ flat pedestal --

def _seam_frame(n=256, bias_level=300.0, seam_step=40.0, illum=1500.0):
    bias = np.full((n, n), bias_level, dtype='float32')
    bias[n // 2 - 1:n // 2 + 1, :] += seam_step
    bias[:, n // 2 - 1:n // 2 + 1] += seam_step
    yy, xx = np.mgrid[0:n, 0:n]
    resp = (1 - 0.25 * (((xx - n / 2) / (n / 2)) ** 2
                        + ((yy - n / 2) / (n / 2)) ** 2)).astype('float32')
    return bias, resp, (bias + resp * illum).astype('float32')


def _row_seam_pct(a):
    """Seam rows against the baseline immediately outside them.

    Using a far-away reference row instead would fold in the smooth vignetting
    gradient, which is real signal and legitimately non-zero; this isolates the
    discontinuity at the quadrant boundary.
    """
    c = a.shape[0] // 2
    seam = a[c - 1:c + 1].mean()
    base = np.concatenate([a[c - 6:c - 2].ravel(), a[c + 2:c + 6].ravel()]).mean()
    return 100 * (seam - base) / base


def test_flat_pedestal_removal_kills_the_sensor_seam():
    """A raw flat carries the bias, so dividing by it re-imprints the quadrant
    seam INVERTED with amplitude seam_step/flat_level. Removing the pedestal
    first must eliminate it."""
    bias, resp, flat1 = _seam_frame(illum=1500.0)
    _, _, flat2 = _seam_frame(illum=2600.0)
    dark = bias + 5.0
    light = (bias + 5.0 + resp * 200).astype('float32')

    raw_master = np.median([flat1, flat2], axis=0)
    old = calib.calibrate(light, dark, None, raw_master)

    prepared = [calib.prepare_flat_frame(f, pedestal=dark) for f in (flat1, flat2)]
    new = calib.calibrate(light, dark, None, np.median(prepared, axis=0))

    assert abs(_row_seam_pct(old)) > 1.0, "test setup should show the bug"
    assert abs(_row_seam_pct(new)) < 0.05, (
        f"seam survived pedestal removal: {_row_seam_pct(new):+.3f}%")


def test_master_dark_works_as_the_flat_pedestal_when_no_bias():
    """With no bias loaded the master dark supplies the same bias structure."""
    bias, resp, flat = _seam_frame(illum=2000.0)
    dark = bias + 5.0
    light = (bias + 5.0 + resp * 200).astype('float32')

    new = calib.calibrate(light, dark, None, calib.prepare_flat_frame(flat, pedestal=dark))
    assert abs(_row_seam_pct(new)) < 0.05


def test_flats_are_scaled_to_unity_before_combining():
    """Twilight flats drift in level; combining them unscaled weights the
    brightest frame (IRAF flatcombine scale=mode, ccdproc scale=inv_median).
    After pedestal removal and unity scaling, flats taken at very different
    levels must collapse onto the same response surface."""
    bias, _, flat_dim = _seam_frame(illum=1000.0)
    _, _, flat_bright = _seam_frame(illum=3000.0)
    dark = bias + 5.0   # the pedestal really present in the flats

    a = calib.prepare_flat_frame(flat_dim, pedestal=dark)
    b = calib.prepare_flat_frame(flat_bright, pedestal=dark)

    assert np.median(a) == pytest.approx(1.0, abs=1e-3)
    assert np.median(b) == pytest.approx(1.0, abs=1e-3)
    # Same response shape despite a 3x level difference.
    np.testing.assert_allclose(a, b, rtol=5e-3)


def test_pedestal_removal_precedes_scaling():
    """Scaling a frame that still carries the bias scales the bias too, so the
    seam survives into the normalised flat."""
    bias, _, flat = _seam_frame(illum=2000.0)
    dark = bias + 5.0

    correct = calib.prepare_flat_frame(flat, pedestal=dark, scale_to_unity=True)
    scaled_only = calib.prepare_flat_frame(flat, pedestal=None, scale_to_unity=True)

    assert abs(_row_seam_pct(correct)) < 0.05, "pedestal removal left a seam"
    assert abs(_row_seam_pct(scaled_only)) > 1.0, "scaling alone should keep the seam"


# ------------------------------------------------------------------ output --

def test_output_uses_the_same_convention_as_the_raw_frames(tmp_path):
    """AstroImageJ (ImageJ) has no signed 16-bit pixel type: it reads BITPIX=16
    as UNSIGNED. A custom BSCALE/BZERO therefore renders a 65 ADU sky as ~59000
    and makes every calibrated frame look like noise. Output must use exactly
    the raw FLI convention: BSCALE=1, BZERO=32768."""
    data = np.random.default_rng(1).normal(200, 30, (64, 64)).astype('float32')
    written = calib.save_fits_16bit(str(tmp_path / "s.fits"), data,
                                    header=fits.Header(), overwrite=True)

    with fits.open(written, do_not_scale_image_data=True) as hdul:
        hdr = hdul[0].header
        assert hdr['BITPIX'] == 16
        assert hdr['BSCALE'] == 1
        assert hdr['BZERO'] == 32768

    # The BZERO=32768 pattern every reader special-cases: adding BZERO to the
    # raw int16 on disk must give the true ADU. (Previously the writer stored
    # small NEGATIVE int16 with an arbitrary BZERO, which ImageJ reinterpreted
    # as huge unsigned values around 59000-65535.)
    with fits.open(written, do_not_scale_image_data=True) as hdul:
        raw = hdul[0].data.astype(np.int32)
    np.testing.assert_allclose(raw + 32768, np.rint(data), atol=0)
    assert raw.min() >= -32768


def test_ceiling_is_65535_not_4095(tmp_path):
    """The point of the writer change: a star in a vignetted corner exceeds the
    raw 4095 ADC ceiling after flat division and must not be clamped."""
    data = np.full((8, 8), 200.0, dtype='float32')
    data[0, 0] = 6120.7
    written = calib.save_fits_16bit(str(tmp_path / "hi.fits"), data,
                                    header=fits.Header(), overwrite=True)
    back = mfc.load_fits(written)
    assert back[0, 0] == pytest.approx(6121.0)


def test_values_round_to_integer_adu(tmp_path):
    data = np.random.default_rng(0).normal(500, 20, (64, 64)).astype('float32')
    written = calib.save_fits_16bit(str(tmp_path / "c.fits"), data,
                                    header=fits.Header(), overwrite=True)
    back = mfc.load_fits(written)
    # Whole ADU: 0.29 ADU rms quantisation against ~3.4 ADU read noise.
    np.testing.assert_allclose(back, np.rint(data), atol=0)


def test_output_stays_two_bytes_per_pixel(tmp_path):
    """Calibrated frames must not double in size versus the raw 12-bit data."""
    data = np.random.default_rng(1).normal(200, 30, (256, 256)).astype('float32')
    written = calib.save_fits_16bit(str(tmp_path / "s.fits"), data,
                                    header=fits.Header(), overwrite=True)
    with fits.open(written, do_not_scale_image_data=True) as hdul:
        assert hdul[0].data.dtype.itemsize == 2


def test_clamped_pixels_are_recorded(tmp_path):
    """Clamping is a real loss, so it must be visible in the header rather
    than silent."""
    data = np.full((8, 8), 200.0, dtype='float32')
    data[0, 0] = -500.0
    written = calib.save_fits_16bit(str(tmp_path / "neg.fits"), data,
                                    header=fits.Header(), overwrite=True)
    history = "\n".join(fits.getheader(written).get('HISTORY', []))
    assert "below 0 ADU clamped" in history
    assert mfc.load_fits(written)[0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------- autoflat --

@pytest.mark.parametrize("deg", [2, 3, 4])
def test_separable_poly_eval_matches_dense_design_matrix(deg):
    """eval_poly_2d replaced a dense ny*nx*n_terms design matrix (1.25 GB for a
    4096x4096 frame at degree 3). It must give the same surface."""
    rng = np.random.default_rng(0)
    _, terms = calib.build_poly_design(np.zeros(1), np.zeros(1), deg)
    coeffs = rng.normal(size=len(terms))

    ny, nx = 97, 123
    yy, xx = np.mgrid[0:ny, 0:nx]
    X_full, terms2 = calib.build_poly_design(((xx - nx / 2) / (nx / 2)).ravel(),
                                             ((yy - ny / 2) / (ny / 2)).ravel(), deg)
    assert terms2 == terms
    dense = (X_full @ coeffs).reshape(ny, nx)

    np.testing.assert_allclose(calib.eval_poly_2d(coeffs, terms, ny, nx, deg),
                               dense, rtol=0, atol=1e-5)


def test_auto_flat_is_normalised_and_follows_the_frame():
    """The auto-flat models THIS frame's sky, so two frames with different
    vignetting must produce different surfaces -- the worker refits per frame
    instead of caching the first one."""
    ny = nx = 256
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = ((xx - nx / 2) / (nx / 2)) ** 2 + ((yy - ny / 2) / (ny / 2)) ** 2

    frame_a = (500.0 * (1 - 0.20 * r2)).astype('float32')
    frame_b = (500.0 * (1 - 0.05 * r2)).astype('float32')

    model_a = calib.create_auto_flat_from_light(frame_a)['model']
    model_b = calib.create_auto_flat_from_light(frame_b)['model']

    assert np.median(model_a) == pytest.approx(1.0, abs=1e-3)
    assert np.median(model_b) == pytest.approx(1.0, abs=1e-3)
    # Different vignetting must not collapse onto the same surface.
    assert np.abs(model_a - model_b).max() > 0.02


def test_stale_bscale_bzero_from_the_source_header_is_dropped(tmp_path):
    """Raw frames carry BSCALE/BZERO; reusing them verbatim would fight with
    the values astropy derives for the uint16 array."""
    src = fits.Header()
    src['BSCALE'] = 0.129
    src['BZERO'] = 840.26
    data = np.full((8, 8), 200.0, dtype='float32')
    written = calib.save_fits_16bit(str(tmp_path / "stale.fits"), data,
                                    header=src, overwrite=True)
    assert mfc.load_fits(written)[0, 0] == pytest.approx(200.0)


def test_saturation_level_is_recorded(tmp_path):
    """Photometry still needs to know the RAW full scale, because after flat
    division the ceiling is no longer 4095."""
    data = np.random.default_rng(2).normal(200, 30, (32, 32)).astype('float32')
    written = calib.save_fits_16bit(str(tmp_path / "sat.fits"), data,
                                    header=fits.Header(), saturate_level=4095,
                                    overwrite=True)
    assert fits.getheader(written)['SATURATE'] == pytest.approx(4095.0)


def test_non_finite_pixels_do_not_poison_the_file(tmp_path):
    data = np.random.default_rng(3).normal(200, 10, (32, 32)).astype('float32')
    data[5, 5] = np.nan
    data[6, 6] = np.inf
    written = calib.save_fits_16bit(str(tmp_path / "nan.fits"), data,
                                    header=fits.Header(), overwrite=True)
    back = mfc.load_fits(written)
    assert np.isfinite(back).all()
