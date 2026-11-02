"""
Camera noise-model configuration -- single source of truth for the detector
parameters used in the CCD/CMOS SNR equation (core.photometry_core).

Target detector: FLI Kepler KL4040 (Gpixel GSENSE4040 sCMOS, dual-gain HDR).

UNIT CONVENTIONS (must match photometry_core's noise equation):

  gain        CONVERSION gain in electrons per ADU (e-/ADU), as measured by
              the photon-transfer method (see tools/measure_gain.py).
              *** This is NOT the "GAIN" value FLIPilot writes into FITS
              headers (16.5 for High, 2.8 for Low, etc.) -- that number is a
              camera amplifier/saving setting, not a conversion gain. ***
              Evidence that High-mode conversion gain is ~1 e-/ADU:
                - INOOR's own photon-transfer measurement: ~1.0 e-/ADU
                - datasheet read noise (3.7 e-) / measured read noise in ADU
                  (3.18 ADU) = 1.16 e-/ADU
                - Moravian C4-16000 (same GSENSE4040 sensor) publishes
                  0.85 e-/ADU (high gain) / 19.5 e-/ADU (low gain)

  read_noise  RMS read noise in ELECTRONS per pixel per read.
              KL4040 datasheet: 3.7 e- (High mode). INOOR measurement from the
              photon-transfer analysis (dark pairs): sqrt(Var(D1-D2)/2) x gain
              = 3.8 e-. An earlier value of 3.18 e-, taken from a plain bias
              difference, was not accurate and has been superseded.

  dark_noise  Dark charge per pixel in ELECTRONS accumulated over the
              exposure (its Poisson variance in e-^2 equals the charge).
              With a known exposure it is dark_current x EXPTIME
              (CameraMode.dark_charge); the fixed 4.15 below is only the
              fallback for an exposure of unknown length.

Confirmation from the camera's own FITS headers (FliPilot 1.2.91, KL2573420):
  H_ADUE   = 1     "High Gain ADUe value used" -> High channel is 1 e-/ADU,
                   exactly matching INOOR's photon-transfer measurement.
  MERGE_GR = 0.11  "Merge GainRatio" -> Low channel responds at 0.11x the
                   High channel, so g_low = H_ADUE / MERGE_GR ~= 9.1 e-/ADU.
  GAIN     = 16.5 (High) / 2.8 (Low) are FliPilot amplifier SETTINGS, not
                   conversion gains, and must never be fed to the SNR equation.
  IMGTYPE  = 1 (High) / 3 (Low) / 5 (Merge), plus ISHIFRM / ISMRGFRM flags,
                   allow automatic mode detection (see detect_mode_from_header).

Merged (HDR) frames are stored on the High-gain ADU scale (~1 e-/ADU across
the full range): below the merge crossover they ARE high-channel data; above
it they are low-channel data rescaled by 1/MERGE_GR. The *scale* gain is
therefore ~1.0, but the effective read noise above the crossover is the low
channel's (30.1 e- vs 3.8 e-, i.e. ~8x higher in electrons). Faint-star
photometry on Merge frames behaves like High mode; bright-star noise is
slightly underestimated.

BINNING: the camera's 2x2 output is k x (sum of the four pixels), k = 0.400
(High) / 0.473 (Low), so 2x2 has its own measured gain and read noise
(KL4040_*_2X2 below), picked from XBINNING/YBINNING by detect_mode_or_none.
Despite ISSFTBIN=T in the header, the 2x2 frame is not computed from the 1x1
digital numbers: an isolated hot pixel stays linear in 2x2 well above the level
where its 1x1 value bends. Charge binning on the sensor is the likely cause, not
proven, and why k has these values is unknown. No other binning has been
measured; for those the caller's configured values are used, with a warning.

Only HIGH is verified; LOW and MERGE values are derived from header keywords
(H_ADUE, MERGE_GR) and must still be confirmed by photon transfer.
"""
import os
import re
from dataclasses import dataclass
from typing import Optional

from utils.logging_config import get_logger

log = get_logger(__name__)


# Thermal (dark) current of the KL4040 at -10 C, e- per pixel per second.
# Measured on the 2025-10-15 High 1x1 dark ramp (29-194 s): 0.048 over the
# whole frame (0.039 in the central region); the FLI manual quotes 0.15 at
# -20 C. Both channels read the same thermal electrons, so one rate serves
# High, Low and Merge. The paper's Table 1 note gives this value.
DARK_CURRENT_E_PER_S = 0.048

# The KL4040 digitises to 12 bits, and calibrated frames saved with a 12-bit
# range are re-clipped here: a pixel at this value has lost signal.
ADC_CEILING_ADU = 4095


@dataclass(frozen=True)
class CameraMode:
    name: str
    gain: float                     # e-/ADU (conversion gain)
    read_noise: float               # e- RMS per pixel
    dark_noise: float               # e-/pix, fallback when the exposure is unknown
    verified: bool = False          # True once photon-transfer measured
    note: str = ""
    dark_current: float = DARK_CURRENT_E_PER_S   # e-/pix/s
    binning: int = 1                # NxN binning these values are for
    # Highest peak (calibrated ADU, one read) still on the linear response;
    # None where it has not been measured (only the ADC ceiling is checked).
    linear_limit_adu: Optional[float] = None

    def dark_charge(self, exptime_s):
        """
        Dark charge per pixel (e-) collected in `exptime_s` seconds; its
        Poisson variance equals the charge. A fixed 4.15 e- was used for every
        exposure: right for ~86 s, but 2.8 e- for a 58 s frame and 14.4 e- for
        300 s. Falls back to dark_noise only when the exposure is unknown.
        """
        if exptime_s is None or not (float(exptime_s) > 0):
            return self.dark_noise
        return self.dark_current * float(exptime_s)


KL4040_HIGH = CameraMode(
    name="High",
    gain=0.93,
    read_noise=3.8,
    dark_noise=4.15,
    verified=True,
    note="Conversion gain from the full photon-transfer curve on dome flats "
         "(2025-12-23, matched-dark subtracted, central 1024x1024): "
         "g = 0.93 +/- 0.03 e-/ADU (weighted fit; through-origin 0.97; i.e. "
         "~0.93-0.97, consistent with 1). Read noise 3.8 e- is a by-product of "
         "the same photon-transfer analysis (dark pairs, "
         "sqrt(Var(D1-D2)/2) x gain), and agrees with the 3.7 e- datasheet "
         "value. It SUPERSEDES the earlier 3.18 e-, which came from a plain "
         "bias difference and was not accurate. "
         "Changing gain only rescales SNR/error bars, never the magnitude. "
         "Linear to 3500 ADU per read: measured on-sky (GRB 251013C r, 26 "
         "calibrated frames, 22 PS1 stars, 2026-08-25) the zeropoint residual "
         "is flat to +/-0.004 mag below 3500 and falls off a cliff above 3850.",
    linear_limit_adu=3500.0,
)

KL4040_LOW = CameraMode(
    name="Low",
    gain=18.7,
    read_noise=30.1,
    dark_noise=4.15,
    verified=True,
    note="Conversion gain from the photon-transfer curve on Low-mode dome "
         "flats (2025-12-23): g = 18.7 +/- 2.3 e-/ADU. This SUPERSEDES the old "
         "header-derived guess (H_ADUE/MERGE_GR = 1/0.11 ~= 9.1), which is "
         "wrong -- MERGE_GR is not the conversion-gain ratio. Read noise 30.1 "
         "e- is a PTC by-product (sqrt(Var(D1-D2)/2) x gain ~= 1.6 ADU x 18.7); "
         "physically the low-gain channel trades read noise for full well, so "
         "a large e- value is expected, but treat 30 e- as approximate.",
)

KL4040_MERGE = CameraMode(
    name="Merge",
    gain=KL4040_HIGH.gain,        # HDR frames sit on the High-gain ADU scale
    read_noise=3.8,
    dark_noise=4.15,
    verified=False,
    note="Merged frames are stored on the High-gain ADU scale. Verified on "
         "these dome flats: Merge is byte-identical to High for the whole "
         "dim range (0 pixels diverge), because the HDR crossover to the "
         "low-gain branch is never reached below the ~4095 12-bit ceiling. So "
         "g = High = 0.93 in the low-signal regime. Above the crossover the "
         "effective read noise becomes the low channel's (30.1 e- vs 3.8 e-, "
         "~8x higher) and the merge gain is NOT constrained by this dataset "
         "-- needs bright flats.",
)

# 2x2 binning. The camera outputs k x (sum of the four pixels), k = 0.400 (High)
# and 0.473 (Low), so the effective gain is about g(1x1)/k: 2.5x the unbinned
# value, not 4x. Binned data
# reduced with the 1x1 values get their error bars wrong by about that factor.
# Photon transfer on the 2026-08-03 dome flats, set B. A binned pixel collects
# the dark current of four. 3x3 rests on one signal level with no read noise, so
# it is not offered here.
KL4040_HIGH_2X2 = CameraMode(
    name="High 2x2",
    gain=2.314,
    read_noise=9.28,
    dark_noise=4 * KL4040_HIGH.dark_noise,
    verified=True,
    dark_current=4 * DARK_CURRENT_E_PER_S,
    binning=2,
    note="g = 2.314 +/- 0.008 e-/ADU, read noise 9.28 e- per binned pixel, 31% "
         "above four independent 1x1 pixels (the camera strips and re-applies a "
         "black level around the binning). Linear to 2548 ADU.",
)

KL4040_LOW_2X2 = CameraMode(
    name="Low 2x2",
    gain=41.56,
    read_noise=67.7,
    dark_noise=4 * KL4040_LOW.dark_noise,
    verified=True,
    dark_current=4 * DARK_CURRENT_E_PER_S,
    binning=2,
    note="g = 41.56 +/- 9.4 e-/ADU (23%: the Low 2x2 flats carry little "
         "signal), read noise 67.7 e- per binned pixel.",
)

KL4040_MERGE_2X2 = CameraMode(
    name="Merge 2x2",
    gain=KL4040_HIGH_2X2.gain,
    read_noise=KL4040_HIGH_2X2.read_noise,
    dark_noise=4 * KL4040_MERGE.dark_noise,
    verified=False,
    dark_current=4 * DARK_CURRENT_E_PER_S,
    binning=2,
    note="Bit-identical to High 2x2 in every 2026-08 frame (the HDR blend never "
         "engaged), so High 2x2's values; unverified above the crossover.",
)

MODES = {m.name: m for m in (KL4040_HIGH, KL4040_LOW, KL4040_MERGE)}
_MODES_2X2 = {"High": KL4040_HIGH_2X2, "Low": KL4040_LOW_2X2, "Merge": KL4040_MERGE_2X2}

# The mode INOOR assumes by default. All observatory data processed so far
# is saved in High mode.
DEFAULT_MODE = KL4040_HIGH


def get_mode(name: Optional[str]) -> CameraMode:
    """Return the CameraMode for `name`, falling back to DEFAULT_MODE."""
    if not name:
        return DEFAULT_MODE
    return MODES.get(str(name).strip().capitalize(), DEFAULT_MODE)


# FliPilot IMGTYPE codes observed in KL4040 headers.
_IMGTYPE_TO_MODE = {1: "High", 3: "Low", 5: "Merge"}


def detect_mode_from_header(hdr) -> CameraMode:
    """
    Detect the camera saving mode from a FliPilot FITS header, using (in
    order of reliability) ISMRGFRM / ISHIFRM flags and the IMGTYPE code
    (1=High, 3=Low, 5=Merge). Falls back to DEFAULT_MODE when the header
    is missing or not from a Kepler camera.
    """
    if hdr is None:
        return DEFAULT_MODE
    try:
        if bool(hdr.get('ISMRGFRM', False)):
            return KL4040_MERGE
        if bool(hdr.get('ISHIFRM', False)):
            return KL4040_HIGH
        mode_name = _IMGTYPE_TO_MODE.get(hdr.get('IMGTYPE'))
        if mode_name:
            return MODES[mode_name]
    except Exception:
        pass
    return DEFAULT_MODE


# FliPilot writes the saving mode into the filename too, e.g.
#   Flat_Clear_..._High_1.fit / ..._Low_1.fit / ..._Merge_1.fit
_FILENAME_MODE_KEYS = (("_merge", "Merge"), ("_low", "Low"), ("_high", "High"))


_BIN_TOKEN = re.compile(r'_(\d+)x(\d+)_')
_warned_binning = set()


def frame_binning(hdr=None, filename=None):
    """(xbin, ybin) from XBINNING/YBINNING, else from the '_2x2_'-style token
    FliPilot writes into the filename, else (1, 1)."""
    if hdr is not None:
        xb, yb = hdr.get('XBINNING'), hdr.get('YBINNING')
        if xb is not None or yb is not None:
            return int(xb or 1), int(yb or 1)
    if filename:
        m = _BIN_TOKEN.search(os.path.basename(str(filename)).lower())
        if m:
            return int(m.group(1)), int(m.group(2))
    return 1, 1


def detect_mode_or_none(hdr=None, filename=None) -> Optional[CameraMode]:
    """
    Confidently detect the camera saving mode from a FITS header and/or a
    filename. Unlike `detect_mode_from_header`, this returns **None** when
    nothing identifies the mode, so the caller can keep its own configured
    fallback instead of silently assuming DEFAULT_MODE.

    Order of trust: header flags/IMGTYPE (most reliable) -> filename suffix.

    The binning picks the values: 1x1 and 2x2 have measured ones. Any other
    binning returns None, with a warning, because nothing has been measured
    for it and the 1x1 values would be wrong by a large factor.
    """
    mode = _unbinned_mode_or_none(hdr, filename)
    if mode is None:
        return None
    xb, yb = frame_binning(hdr, filename)
    if (xb, yb) == (1, 1):
        return mode
    if (xb, yb) == (2, 2):
        return _MODES_2X2[mode.name]
    key = (str(filename), xb, yb)
    if key not in _warned_binning:
        _warned_binning.add(key)
        log.warning("%s: no measured noise parameters for %dx%d binning; "
                    "the configured gain/read noise are used instead",
                    os.path.basename(str(filename)), xb, yb)
    return None


def _unbinned_mode_or_none(hdr=None, filename=None) -> Optional[CameraMode]:
    """The 1x1 mode named by the header flags/IMGTYPE or the filename."""
    if hdr is not None:
        try:
            if bool(hdr.get('ISMRGFRM', False)):
                return KL4040_MERGE
            if bool(hdr.get('ISHIFRM', False)):
                return KL4040_HIGH
            mode_name = _IMGTYPE_TO_MODE.get(hdr.get('IMGTYPE'))
            if mode_name:
                return MODES[mode_name]
        except Exception:
            pass
    if filename:
        base = os.path.basename(str(filename)).lower()
        for key, mode_name in _FILENAME_MODE_KEYS:
            if key in base:
                return MODES[mode_name]
    return None
