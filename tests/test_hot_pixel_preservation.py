"""
A master bias/dark must keep its hot pixels -- removing them from the
science frame is the master's entire job.

The 2D sigma-clip post-filter replaces spatial outliers with the median of
their block, so it erases those pixels. Measured on a real 6-frame master
bias (kernel 35, auto, upper 3.0): three pixels at the end of row 4095
sitting at 1400-1900 ADU in ALL SIX frames were flattened to 98 ADU, which
leaves a +1486 ADU spike in every frame the master is subtracted from. The
same run rewrote 0.184% of the master while pure Gaussian noise of the same
width gave 0.144%, i.e. most of what it removes is noise, not defects.

These tests pin the default (off) and the consequence of turning it on.
"""
import numpy as np
import pytest
from astropy.io import fits

from core import masterFrame_creator as mfc
from models.file_set import FileSet


def write_fits(path, data):
    fits.PrimaryHDU(np.asarray(data, dtype='float32')).writeto(str(path))
    return str(path)


def bias_with_hot_pixel(tmp_path, n=6, level=94.0, hot=1500.0, seed=1):
    """n bias frames sharing one stable hot pixel at (40, 40)."""
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        frame = rng.normal(level, 4.0, (80, 80))
        frame[40, 40] = hot + rng.normal(0, 50)   # stable in every frame
        paths.append(write_fits(tmp_path / f"bias_{i}.fits", frame))
    return paths


def test_clipping_is_off_by_default():
    assert FileSet().do_sigma_clip is False, (
        "the post-filter erases hot pixels from masters; it must be opt-in")


def test_hot_pixel_survives_the_default_path(tmp_path):
    paths = bias_with_hot_pixel(tmp_path)
    master, hdr = mfc.create_master(None, paths, method='median')
    assert hdr['STK_CLIP'] is False
    assert master[40, 40] > 1200, (
        f"stable hot pixel was lost: {master[40, 40]}")


def test_clipping_erases_the_hot_pixel_and_says_so(tmp_path):
    """If the user turns it on, the damage must at least be recorded."""
    paths = bias_with_hot_pixel(tmp_path)
    master, hdr = mfc.create_master(None, paths, method='median',
                                    do_sigma_clip=True, sigma_upper=3.0,
                                    kernel=35, auto_sigma=True)
    assert master[40, 40] < 200, (
        "this test documents the erasure; if it no longer happens the "
        "filter changed and the warnings around it need revisiting")
    assert hdr['STK_CLIP'] is True
    assert hdr['STK_NREP'] >= 1
    assert isinstance(hdr['STK_NREP'], int)


def test_replacement_count_is_zero_when_nothing_is_clipped(tmp_path):
    flat = [write_fits(tmp_path / f"f{i}.fits", np.full((40, 40), 100.0))
            for i in range(3)]
    _, hdr = mfc.create_master(None, flat, method='median',
                               do_sigma_clip=True, sigma_upper=3.0,
                               kernel=35, auto_sigma=True)
    assert hdr['STK_NREP'] == 0
