"""
Guards inside CalibrationWorker's light-frame loop.

A light frame with no EXPTIME cannot have a dark scaled to it, and the worker
means to say so. It read `dm` to decide -- but `dm` is only assigned forty
lines further down, so on the FIRST such frame Python raised
`UnboundLocalError` instead, and on a later one it silently consulted the
PREVIOUS frame's dark. The test drives the real run() to the point of failure
and checks the message the operator actually sees.
"""
import numpy as np
import pytest
from astropy.io import fits

from controllers.workers import CalibrationWorker


def _write(path, value, shape=(16, 16), exptime=None):
    hdr = fits.Header()
    if exptime is not None:
        hdr['EXPTIME'] = exptime
    fits.PrimaryHDU(np.full(shape, value, dtype='float32'), header=hdr).writeto(path)
    return str(path)


def _run_and_capture(worker):
    """Run the worker body in this thread and return its (ok, message)."""
    out = {}
    worker.finished.connect(lambda ok, msg: out.update(ok=ok, msg=msg))
    worker.run()
    return out.get('ok'), out.get('msg')


def test_light_without_exptime_reports_instead_of_crashing(tmp_path):
    sets = [
        {'type': 'Light', 'files': [_write(tmp_path / "light.fits", 500.0)],
         'bad_indices': set()},
        {'type': 'Dark', 'files': [_write(tmp_path / "dark.fits", 10.0, exptime=30.0)],
         'bad_indices': set()},
        {'type': 'Flat', 'files': [_write(tmp_path / "flat.fits", 1000.0, exptime=1.0)],
         'bad_indices': set()},
    ]
    ok, msg = _run_and_capture(
        CalibrationWorker(sets, str(tmp_path), scale_dark=True, skip_bias=True))

    assert ok is False
    assert "no exposure time" in msg, (
        "the operator must be told which frame lacks an exposure time; "
        f"got {msg!r}")
    assert "dm" not in msg, f"leaked an UnboundLocalError to the user: {msg!r}"


def test_no_scaling_requested_tolerates_a_missing_exptime(tmp_path):
    """With dark scaling off the exposure time is not needed, so the run
    proceeds rather than refusing."""
    sets = [
        {'type': 'Light', 'files': [_write(tmp_path / "light.fits", 500.0)],
         'bad_indices': set()},
        {'type': 'Dark', 'files': [_write(tmp_path / "dark.fits", 10.0, exptime=30.0)],
         'bad_indices': set()},
        {'type': 'Flat', 'files': [_write(tmp_path / "flat.fits", 1000.0, exptime=1.0)],
         'bad_indices': set()},
    ]
    ok, msg = _run_and_capture(
        CalibrationWorker(sets, str(tmp_path), scale_dark=False, skip_bias=True))

    assert ok is True, f"run should have completed; failed with {msg!r}"


def test_clamped_pixels_are_reported_to_the_operator(tmp_path):
    """
    The 16-bit output cannot store a negative pixel, so an over-subtracted one
    is written as 0 and stops being data. On real INO340 frames this ran to
    ~10,000 pixels per frame, and the operator was told only "Calibration
    complete" -- the count lived in a HISTORY line and a log warning.
    """
    # A handful of hot pixels in the dark, as on the real detector: the frame
    # stays good ADU data, but those few pixels go negative.
    light = np.full((16, 16), 900.0, dtype='float32')
    dark = np.full((16, 16), 100.0, dtype='float32')
    dark[0, :5] = 3000.0                       # 5 hot pixels
    hdr = fits.Header(); hdr['EXPTIME'] = 10.0
    fits.PrimaryHDU(light, header=hdr).writeto(tmp_path / "light.fits")
    fits.PrimaryHDU(dark, header=hdr).writeto(tmp_path / "dark.fits")

    sets = [
        {'type': 'Light', 'files': [str(tmp_path / "light.fits")], 'bad_indices': set()},
        {'type': 'Dark', 'files': [str(tmp_path / "dark.fits")], 'bad_indices': set()},
        {'type': 'Flat', 'files': [_write(tmp_path / "flat.fits", 1000.0, exptime=1.0)],
         'bad_indices': set()},
    ]
    w = CalibrationWorker(sets, str(tmp_path), scale_dark=False, skip_bias=True)
    ok, msg = _run_and_capture(w)

    assert ok is True
    assert "WARNING" in msg and "below 0 ADU" in msg, (
        f"clamping must reach the completion dialog; got {msg!r}")
    assert w.clamp_report and w.clamp_report[0][1] == 5, (
        f"should have counted the 5 hot pixels; got {w.clamp_report}")


def test_a_clean_run_says_nothing_about_clamping(tmp_path):
    sets = [
        {'type': 'Light', 'files': [_write(tmp_path / "light.fits", 900.0, exptime=10.0)],
         'bad_indices': set()},
        {'type': 'Dark', 'files': [_write(tmp_path / "dark.fits", 100.0, exptime=10.0)],
         'bad_indices': set()},
        {'type': 'Flat', 'files': [_write(tmp_path / "flat.fits", 1000.0, exptime=1.0)],
         'bad_indices': set()},
    ]
    ok, msg = _run_and_capture(
        CalibrationWorker(sets, str(tmp_path), scale_dark=False, skip_bias=True))
    assert ok is True
    assert "WARNING" not in msg, f"nothing clamped, so say nothing: {msg!r}"
