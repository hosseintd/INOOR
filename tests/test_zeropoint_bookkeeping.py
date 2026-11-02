"""
Per-file zeropoints must survive, and the aperture correction with them.

Two formulas used to compute the same magnitude. `set_zeropoint_for_file` used
the stored instrumental magnitude, which carries the frame's aperture
correction. `_update_all_frame_results` instead rebuilt
`-2.5*log10(flux/exptime)` from the raw aperture flux -- which does not -- and
added the GLOBAL zeropoint on top, never consulting model.zeropoints.

So touching the zeropoint or exposure box wiped every per-file calibration and
silently dropped the aperture correction (~0.27 mag at 1 FWHM). It fired after
every bulk run too, because on_bulk_done set the exposure box without blocking
signals.
"""
import math

import pytest

from models.multi_photometry_model import MultiPhotometryModel


AP_CORR = -0.27          # what a 1-FWHM aperture costs on INO340 frames
FLUX, EXPTIME = 50000.0, 58.14
INSTR = -2.5 * math.log10(FLUX / EXPTIME) + AP_CORR


def model_with_two_frames():
    m = MultiPhotometryModel()
    m.add_files(["/data/a.fits", "/data/b.fits"])
    for i in (0, 1):
        d = m._frame_data[i]
        d.selected_source = (100.0, 100.0)
        d.exptime = EXPTIME
        d.aperture_result = {
            'flux': FLUX,
            'instr_mag': INSTR,
            'aperture_correction': AP_CORR,
        }
        d.mag = INSTR
    return m


def test_per_file_zeropoint_beats_the_global_default():
    m = model_with_two_frames()
    m.zeropoint = 20.0                       # global default
    m.set_zeropoint_for_file("a.fits", 25.0)  # a.fits is calibrated

    assert m.zeropoint_for("a.fits") == 25.0
    assert m.zeropoint_for("b.fits") == 20.0
    assert m.magnitude_for(m._frame_data[0]) == pytest.approx(INSTR + 25.0)
    assert m.magnitude_for(m._frame_data[1]) == pytest.approx(INSTR + 20.0)


def test_zeropoint_lookup_accepts_a_full_path():
    """The one writer stored a full path while every reader used a basename."""
    m = model_with_two_frames()
    m.zeropoint = 20.0
    m.set_zeropoint_for_file("a.fits", 25.0)
    assert m.zeropoint_for("/some/other/dir/a.fits") == 25.0


def test_recomputed_magnitude_keeps_the_aperture_correction():
    """
    Rebuilding from flux must add the correction back.

    Without it the reported magnitude jumps by |AP_CORR| the moment the user
    touches the zeropoint or exposure box -- a 270 mmag step on measurements
    whose error bars are a few mmag.
    """
    m = model_with_two_frames()
    m.zeropoint = 20.0
    recomputed = m.magnitude_for(m._frame_data[0])
    assert recomputed == pytest.approx(INSTR + 20.0)
    # and it must NOT equal the uncorrected version
    uncorrected = -2.5 * math.log10(FLUX / EXPTIME) + 20.0
    assert abs(recomputed - uncorrected) == pytest.approx(abs(AP_CORR))


def test_exptime_change_still_takes_effect():
    """The recompute exists so a corrected exposure time propagates."""
    m = model_with_two_frames()
    m.zeropoint = 20.0
    before = m.magnitude_for(m._frame_data[0])
    m._frame_data[0].exptime = EXPTIME * 2.0
    after = m.magnitude_for(m._frame_data[0])
    assert after - before == pytest.approx(2.5 * math.log10(2.0))


def test_missing_flux_or_exptime_gives_no_magnitude():
    """A failed measurement must not produce a plausible-looking number."""
    m = model_with_two_frames()
    m.zeropoint = 20.0
    m._frame_data[0].aperture_result = {'flux': 0.0}
    assert m.magnitude_for(m._frame_data[0]) is None
    m._frame_data[1].exptime = None
    m.exptime_override = None
    assert m.magnitude_for(m._frame_data[1]) is None


def test_the_two_magnitude_formulas_now_agree():
    """
    `set_zeropoint_for_file` applies the zeropoint to the STORED instrumental
    magnitude; `magnitude_for` rebuilds it from flux. Those were the two
    formulas that disagreed. They must now give the same answer.
    """
    m = model_with_two_frames()
    m.zeropoint = 20.0
    m.set_zeropoint_for_file("a.fits", 25.0)
    m.set_zeropoint_for_file("b.fits", 21.5)

    for row in m.get_photometry_table():
        data = m._frame_data[row.index]
        assert row.zeropoint == m.zeropoint_for(data.fname)
        assert m.magnitude_for(data) == pytest.approx(data.mag), (
            "the stored-instr_mag path and the recompute-from-flux path "
            "must not drift apart again")
