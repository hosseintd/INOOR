"""
Storage format of a calibration run.

The 16-bit writer clamps everything below 0 ADU to 0. On the low-gain channel
with a dark sky that is the negative half of the sky noise: PSR J2215 g Low
calibrates to 2.30 +- 1.63 ADU and 15% of its sky pixels were stored as 0,
moving faint stars by 10-30 mmag. A run is therefore stored as 32-bit float
when the sky of ANY of its frames sits less than FLOAT_STORAGE_Z sigma above
0 ADU, and as 16-bit integers (unchanged) otherwise -- one format per run.
"""
import glob
import os

import numpy as np
import pytest
from astropy.io import fits

from core import calibration as calib
from controllers.workers import CalibrationWorker


def _sky(shape, level, sigma, seed):
    return np.random.default_rng(seed).normal(level, sigma, shape)


def test_sky_clip_margin_reads_the_sky_not_the_bad_pixels():
    low = _sky((400, 400), 2.30, 1.63, 1)
    z, sky, sig = calib.sky_clip_margin(low)
    assert z == pytest.approx((2.30 + 0.5) / 1.63, rel=0.03)
    assert z < calib.FLOAT_STORAGE_Z

    high = _sky((400, 400), 41.0, 8.6, 2)
    z_clean = calib.sky_clip_margin(high)[0]
    assert z_clean > calib.FLOAT_STORAGE_Z
    # Dead and hot pixels clamp in almost every frame; they must not decide.
    dirty = high.copy()
    rng = np.random.default_rng(3)
    dirty.flat[rng.choice(dirty.size, 1600, replace=False)] = -500.0
    dirty.flat[rng.choice(dirty.size, 1600, replace=False)] = 60000.0
    assert calib.sky_clip_margin(dirty)[0] == pytest.approx(z_clean, rel=0.02)


def test_float_writer_keeps_negatives_and_fractions(tmp_path):
    data = np.array([[-3.25, 0.4], [2.30, 4100.7]], dtype='float32')
    hdr = fits.Header()
    hdr['BZERO'] = 32768
    hdr['BSCALE'] = 1
    hdr['DATE-OBS'] = '2024-09-09T23:24:47'
    written = calib.save_fits_float32(str(tmp_path / "f.fits"), data, header=hdr,
                                      reason="test reason")
    with fits.open(written) as h:
        assert h[0].header['BITPIX'] == -32
        assert 'BZERO' not in h[0].header
        assert h[0].header['SATURATE'] == 4095.0
        assert "test reason" in str(h[0].header['HISTORY'])
        np.testing.assert_array_equal(h[0].data, data)
    # Same file name as the 16-bit writer gives the same frame.
    same = calib.save_fits_16bit(str(tmp_path / "f.fits"), data + 10.0, header=hdr,
                                 overwrite=True)
    assert same == written


def _write(path, data, exptime=10.0):
    hdr = fits.Header()
    hdr['EXPTIME'] = exptime
    fits.PrimaryHDU(np.asarray(data, dtype='float32'), header=hdr).writeto(path)
    return str(path)


def _run(tmp_path, skies):
    """Calibrate one light per (level, sigma) in `skies`; dark = 100 ADU, flat = flat."""
    shape = (64, 64)
    lights = [_write(tmp_path / f"light{k}.fits", 100.0 + _sky(shape, lv, sg, 10 + k))
              for k, (lv, sg) in enumerate(skies)]
    sets = [
        {'type': 'Light', 'files': lights, 'bad_indices': set()},
        {'type': 'Dark', 'files': [_write(tmp_path / "dark.fits", np.full(shape, 100.0))],
         'bad_indices': set()},
        {'type': 'Flat', 'files': [_write(tmp_path / "flat.fits", np.full(shape, 1000.0), 1.0)],
         'bad_indices': set()},
    ]
    out = tmp_path / "out"
    out.mkdir()
    w = CalibrationWorker(sets, str(out), scale_dark=False, skip_bias=True)
    res = {}
    w.finished.connect(lambda ok, msg: res.update(ok=ok, msg=msg))
    w.run()
    files = sorted(glob.glob(os.path.join(str(out), "Calibrated_*")))
    assert len(files) == len(skies)
    return res, [fits.getheader(f)['BITPIX'] for f in files], files


def test_a_normal_sky_stays_16_bit(tmp_path):
    res, bitpix, files = _run(tmp_path, [(41.0, 8.6), (45.0, 8.6)])
    assert res['ok'] is True
    assert bitpix == [16, 16]
    assert "float" not in res['msg']


def test_a_sky_near_zero_is_stored_as_float(tmp_path):
    res, bitpix, files = _run(tmp_path, [(2.3, 1.63)])
    assert res['ok'] is True
    assert bitpix == [-32]
    assert "32-bit float" in res['msg']
    data = fits.getdata(files[0])
    assert (data < 0).mean() > 0.05, "the negative half of the sky noise must survive"
    assert "WARNING" not in res['msg'], "a float file clamps nothing"


def test_one_low_frame_makes_the_whole_run_float(tmp_path):
    res, bitpix, files = _run(tmp_path, [(41.0, 8.6), (2.3, 1.63), (45.0, 8.6)])
    assert res['ok'] is True
    assert bitpix == [-32, -32, -32], "one format per run"
    assert "All 3 frame(s)" in res['msg']
