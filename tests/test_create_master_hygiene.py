"""
Create Master tab: shape checking and time base.

Both behaviours below were found by driving the running GUI against real
INO340 frames:

  * mixing 2048x2048 and 4096x4096 frames failed only after every frame had
    been read into RAM, with a numpy message naming an array index;
  * the master's CRE_TIME was local time while every other date INOOR writes
    was UT.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from astropy.io import fits

from core import masterFrame_creator as mfc


def write_fits(path, data):
    fits.PrimaryHDU(np.asarray(data, dtype='float32')).writeto(str(path))
    return str(path)


# --- shape pre-flight -------------------------------------------------

def test_mismatched_shapes_name_the_offending_file(tmp_path):
    big = [write_fits(tmp_path / f"big_{i}.fits", np.full((64, 64), 100.0))
           for i in range(3)]
    odd = write_fits(tmp_path / "odd_one.fits", np.full((32, 32), 100.0))

    with pytest.raises(ValueError) as exc:
        mfc.create_master(None, big + [odd], method='median')

    msg = str(exc.value)
    assert "odd_one.fits" in msg, f"the message must name the file: {msg}"
    assert "64x64" in msg and "32x32" in msg


def test_shape_check_runs_before_any_pixel_is_read(tmp_path, monkeypatch):
    good = write_fits(tmp_path / "a.fits", np.full((64, 64), 100.0))
    odd = write_fits(tmp_path / "b.fits", np.full((32, 32), 100.0))

    calls = []
    real_load = mfc.load_fits
    monkeypatch.setattr(mfc, "load_fits",
                        lambda p: (calls.append(p), real_load(p))[1])

    with pytest.raises(ValueError):
        mfc.create_master(None, [good, odd], method='median')
    assert calls == [], "frames must not be loaded before the shape check"


def test_matching_shapes_still_combine(tmp_path):
    paths = [write_fits(tmp_path / f"f{i}.fits", np.full((32, 32), 100.0 + i))
             for i in range(3)]
    master, _ = mfc.create_master(None, paths, method='median')
    assert master.shape == (32, 32)
    assert np.allclose(master, 101.0)


def test_frame_transform_bypasses_the_shape_check(tmp_path):
    """Binning exists precisely to bring frames to a common shape."""
    a = write_fits(tmp_path / "a.fits", np.full((64, 64), 100.0))
    b = write_fits(tmp_path / "b.fits", np.full((32, 32), 100.0))
    master, _ = mfc.create_master(None, [a, b], method='median',
                                  frame_transform=lambda img: img[:32, :32])
    assert master.shape == (32, 32)


# --- time base --------------------------------------------------------

def test_cre_time_is_ut(tmp_path):
    paths = [write_fits(tmp_path / f"f{i}.fits", np.full((16, 16), 100.0))
             for i in range(2)]
    _, hdr = mfc.create_master(None, paths, method='median')
    stamp = datetime.strptime(hdr['CRE_TIME'], "%Y-%m-%dT%H:%M:%S")
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs(now_utc - stamp) < timedelta(minutes=5), (
        f"CRE_TIME {hdr['CRE_TIME']} is not UT (UT now is {now_utc})")
    assert 'UT' in hdr.comments['CRE_TIME']
