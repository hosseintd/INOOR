"""
Guards the camera noise-model configuration:
  - the default mode is the verified High mode with conversion gain ~1 e-/ADU
    (photon-transfer measured), NOT the FITS-header GAIN setting (16.5);
  - photometry_core's function defaults actually come from camera_config, so
    a future edit can't silently reintroduce scattered hardcoded constants.
"""
import inspect

from core import camera_config as cc
from core import photometry_core as pc


def test_default_mode_is_verified_high_mode():
    assert cc.DEFAULT_MODE is cc.KL4040_HIGH
    assert cc.DEFAULT_MODE.verified
    # Conversion gain must be the photon-transfer value (~1), never the
    # FITS-header amplifier setting (16.5).
    assert cc.DEFAULT_MODE.gain < 5.0


def test_photometry_defaults_come_from_camera_config():
    for func in (pc.perform_aperture_photometry, pc.compute_snr_vs_radius):
        params = inspect.signature(func).parameters
        assert params['gain'].default == cc.DEFAULT_MODE.gain
        assert params['read_noise'].default == cc.DEFAULT_MODE.read_noise
        assert params['dark_noise'].default == cc.DEFAULT_MODE.dark_noise


def test_get_mode_lookup_and_fallback():
    assert cc.get_mode('High') is cc.KL4040_HIGH
    assert cc.get_mode('low') is cc.KL4040_LOW
    assert cc.get_mode('MERGE') is cc.KL4040_MERGE
    assert cc.get_mode(None) is cc.DEFAULT_MODE
    assert cc.get_mode('nonexistent') is cc.DEFAULT_MODE


def test_low_gain_is_photon_transfer_measured():
    # g_low is now the DIRECT photon-transfer measurement (dome flats,
    # 2025-12-23): 18.7 +/- 2.3 e-/ADU. This supersedes the old header-derived
    # guess (H_ADUE/MERGE_GR = 1/0.11 ~= 9.1), which was wrong. Guard against a
    # regression back to that placeholder.
    assert abs(cc.KL4040_LOW.gain - 18.7) < 2.5
    assert cc.KL4040_LOW.gain > 12.0            # definitively not the old ~9.1
    # Merged frames are stored on the High-gain ADU scale (low-signal regime).
    assert cc.KL4040_MERGE.gain == cc.KL4040_HIGH.gain


def test_detect_mode_or_none_returns_none_when_unknown():
    from astropy.io import fits
    # No mode info anywhere -> None (so callers keep their configured fallback,
    # instead of silently assuming High like detect_mode_from_header does).
    assert cc.detect_mode_or_none(None, None) is None
    assert cc.detect_mode_or_none(fits.Header(), "some_science_frame.fits") is None
    # Header flags win.
    h = fits.Header(); h['ISMRGFRM'] = True
    assert cc.detect_mode_or_none(h, None) is cc.KL4040_MERGE
    # Filename suffix is the fallback when the header is silent.
    assert cc.detect_mode_or_none(None, "Flat_..._Low_1.fit") is cc.KL4040_LOW
    assert cc.detect_mode_or_none(None, r"C:\d\Flat_..._High_1.fit") is cc.KL4040_HIGH
    assert cc.detect_mode_or_none(None, "x_Merge_1.fit") is cc.KL4040_MERGE


def test_detect_mode_from_header():
    from astropy.io import fits

    def hdr(**cards):
        h = fits.Header()
        for k, v in cards.items():
            h[k] = v
        return h

    # Real KL4040 header fingerprints (from observatory frames, 2025-10-08):
    high = hdr(IMGTYPE=1, ISHIFRM=True, ISMRGFRM=False, GAIN=16.5)
    low = hdr(IMGTYPE=3, ISHIFRM=False, ISMRGFRM=False, GAIN=2.8)
    merge = hdr(IMGTYPE=5, ISHIFRM=False, ISMRGFRM=True, GAIN=16.5)

    assert cc.detect_mode_from_header(high) is cc.KL4040_HIGH
    assert cc.detect_mode_from_header(low) is cc.KL4040_LOW
    assert cc.detect_mode_from_header(merge) is cc.KL4040_MERGE

    # Flags take priority; IMGTYPE alone also works
    assert cc.detect_mode_from_header(hdr(IMGTYPE=3)) is cc.KL4040_LOW
    # Non-Kepler / missing header falls back to the default mode
    assert cc.detect_mode_from_header(hdr(OBJECT='M31')) is cc.DEFAULT_MODE
    assert cc.detect_mode_from_header(None) is cc.DEFAULT_MODE
