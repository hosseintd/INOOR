"""
Tests for the unified EXPTIME parser (utils.exptime_utils) -- previously two
divergent heuristics lived in exptime_utils and multi_photometry_worker and
could disagree about the exposure time of the same file.
"""
import pytest
from astropy.io import fits

from utils.exptime_utils import get_exptime_seconds, format_exptime


def hdr(key='EXPTIME', value=None, comment=None):
    h = fits.Header()
    if value is not None:
        if comment:
            h[key] = (value, comment)
        else:
            h[key] = value
    return h


def test_real_kl4040_header_ticks_via_comment():
    # Exact form from observatory frames: 290697 ticks = 2.90697 s
    h = hdr(value=290697, comment='Duration of exposure in 10us periods')
    assert get_exptime_seconds(h) == pytest.approx(2.90697)


def test_large_value_heuristic_without_comment():
    # 968992 ticks -> ~9.69 s even with no comment
    h = hdr(value=968992)
    assert get_exptime_seconds(h) == pytest.approx(9.68992)


def test_plain_seconds_left_untouched():
    h = hdr(value=30.0, comment='Exposure time in seconds')
    assert get_exptime_seconds(h) == pytest.approx(30.0)


def test_alternate_keywords():
    for key in ('EXP_TIME', 'EXPOSURE', 'EXPT', 'TEXPTIME', 'EXPOS'):
        h = hdr(key=key, value=12.5)
        assert get_exptime_seconds(h) == pytest.approx(12.5), key


def test_missing_keyword_returns_default():
    h = fits.Header()
    h['OBJECT'] = 'M31'
    assert get_exptime_seconds(h) == 1.0            # legacy default
    assert get_exptime_seconds(h, default=None) is None  # worker-style probe
    assert get_exptime_seconds(None, default=None) is None


def test_format_exptime():
    assert format_exptime(hdr(value=3, comment='10us periods')) == '30µs'
    assert format_exptime(hdr(value=30.0)) == '30s'
    assert format_exptime(fits.Header()) == 'N/A'
