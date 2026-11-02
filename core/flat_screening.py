"""Quality screening for a set of flat frames.

Not every frame in a flat folder is a flat. Twilight sequences run past the
linear range at one end and into the stars at the other, and a sensor glitch
can leave a frame that simply does not look like a flat. Combining those into a
master quietly corrupts every science frame divided by it.

Four independent tests, chosen so each one catches a different failure:

1. LEVEL -- a flat is only useful inside the detector's linear range. The
   KL4040's response is flat to ~3500 ADU per sub-exposure and falls off a
   cliff above ~3850 (measured on-sky, see the linearity study), so anything
   brighter carries a systematic the flat field would stamp onto the science
   frames. At the other end a frame near the ~92 ADU pedestal has almost no
   signal and contributes noise rather than response.

2. SATURATION -- pixels pinned at the 12-bit ceiling record no response at all.

3. SPIKES -- stars, satellites and cosmic rays. Counted as pixels far above
   their own local median, so the smooth vignetting pattern does not register.
   Judged RELATIVE to the rest of the set: a dawn twilight flat legitimately
   holds a few stars, and what matters is a frame holding far more than its
   siblings.

4. CONSENSUS -- the catch-all for "this does not look like a flat". Each frame
   is scaled to unity and compared with the per-pixel median of the whole set.
   A good frame differs from that consensus by photon noise; a glitched one
   differs by structure. This is the only test that needs no threshold chosen
   in advance, and it is the one that catches failures nobody anticipated.

   It is measured TWO ways, because one number is not enough. A robust scatter
   (MAD) reports how noisy the difference is, but it is blind to a defect that
   covers less than half the frame -- a dead quadrant is exactly 25%, and the
   MAD of a field that is three-quarters zero stays near zero. So the frame is
   also cut into a 16x16 grid and the largest block-median deviation is taken.
   That number answers "is any PART of this frame at the wrong level", which is
   what a sensor glitch, a shadow or a moved dust donut actually looks like.

LIMITATION worth stating plainly: tests 3 and 4 compare frames WITH EACH OTHER.
If every frame in the set is bad in the same way, they agree with each other
and nothing is flagged. Tests 1 and 2 are absolute and still apply.
"""
import os

import numpy as np
from scipy.ndimage import median_filter

from utils.logging_config import get_logger

log = get_logger(__name__)

# Measured on-sky: response is linear to ~3500 ADU per sub-exposure and falls
# off a cliff above ~3850. Staying under 3500 keeps the flat in the regime the
# science frames are calibrated against.
LINEARITY_LIMIT_ADU = 3500.0

# Below this a frame is almost all pedestal and read noise.
MIN_LEVEL_ADU = 200.0

# The 12-bit ADC ceiling of the raw frames.
SATURATION_ADU = 4095.0
MAX_SATURATED_FRACTION = 1e-4        # 0.01% of the frame

# Relative tests: how far above the set's own median a frame may sit.
SPIKE_RATIO = 5.0
CONSENSUS_RATIO = 3.0
BLOCK_RATIO = 4.0
# Floor for the localised test. A block sitting 0.3% off the consensus is a
# 0.3% photometric error stamped on every science frame divided by the master,
# which is already comparable to the ~4 mmag systematic floor of this system.
MIN_BLOCK_DEVIATION = 0.003

_BIN = 8                             # consensus works on a 512x512 view
_PATCH = 1024                        # central patch used for the spike count
_GRID = 16                           # 16x16 blocks for the localised test


def _max_block_deviation(d, grid=_GRID):
    """Largest |median| over a grid of blocks -- localised level errors.

    Each block of the binned difference holds ~65,000 raw pixels, so its median
    is good to about 1e-4 in relative terms. Anything well above that means a
    REGION of the frame sits at the wrong level, which no amount of global
    scatter would reveal.
    """
    ny, nx = d.shape
    by, bx = ny // grid, nx // grid
    if by < 1 or bx < 1:
        return float(abs(np.median(d)))
    core = d[:by * grid, :bx * grid].reshape(grid, by, grid, bx)
    block_medians = np.median(core, axis=(1, 3))
    return float(np.max(np.abs(block_medians)))


def _central_patch(a, size=_PATCH):
    ny, nx = a.shape
    y0, x0 = max(0, (ny - size) // 2), max(0, (nx - size) // 2)
    return a[y0:y0 + size, x0:x0 + size]


def _binned(a, f=_BIN):
    ny, nx = a.shape
    ny, nx = (ny // f) * f, (nx // f) * f
    return a[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


def _robust_sigma(a):
    med = np.median(a)
    return float(1.4826 * np.median(np.abs(a - med)))


def _spike_count(patch):
    """Pixels far above their own local median -- stars, satellites, cosmics.

    A 5x5 median tracks the smooth illumination and removes it, so only compact
    excursions survive. The threshold is 8 sigma of the residual itself, which
    makes the count independent of the frame's brightness.
    """
    local = median_filter(patch, size=5, mode='mirror')
    resid = patch - local
    sigma = _robust_sigma(resid)
    if sigma <= 0:
        return 0
    return int((resid > 8.0 * sigma).sum())


def screen_flat_frames(paths, load_fits, progress_callback=None,
                       linearity_limit=LINEARITY_LIMIT_ADU,
                       min_level=MIN_LEVEL_ADU):
    """Measure every flat in `paths` and say which ones should not be combined.

    `load_fits` is injected rather than imported so the caller controls how
    frames are read (and so this stays testable without FITS files on disk).

    Returns a dict with 'frames' (one record per input, in order), 'thresholds'
    and 'summary'. Each record carries every measured number, not just the
    verdict, so the operator can disagree with the thresholds and see why.
    """
    n = len(paths)
    if n == 0:
        return {'frames': [], 'thresholds': {}, 'summary': 'No frames to screen.'}

    records = []
    small = []
    for i, p in enumerate(paths):
        if progress_callback:
            progress_callback(i / max(1, n), f"Reading {os.path.basename(p)}")
        rec = {'path': p, 'name': os.path.basename(p), 'reasons': [],
               'median': float('nan'), 'sat_fraction': float('nan'),
               'spikes': -1, 'consensus_sigma': float('nan'),
               'block_deviation': float('nan'), 'ok': True}
        try:
            a = np.asarray(load_fits(p), dtype='float32')
        except Exception as e:
            rec['ok'] = False
            rec['reasons'].append(f"cannot be read ({e})")
            records.append(rec)
            small.append(None)
            continue

        rec['median'] = float(np.median(a))
        rec['sat_fraction'] = float((a >= SATURATION_ADU).mean())
        rec['spikes'] = _spike_count(_central_patch(a))
        small.append(_binned(a))
        records.append(rec)

    # --- absolute tests -----------------------------------------------------
    for rec in records:
        if not np.isfinite(rec['median']):
            continue
        if rec['median'] > linearity_limit:
            rec['reasons'].append(
                f"level {rec['median']:.0f} ADU is above the {linearity_limit:.0f} ADU "
                f"linearity limit")
        if rec['median'] < min_level:
            rec['reasons'].append(
                f"level {rec['median']:.0f} ADU is too low to carry useful signal")
        if rec['sat_fraction'] > MAX_SATURATED_FRACTION:
            rec['reasons'].append(
                f"{100 * rec['sat_fraction']:.3f}% of pixels are at the "
                f"{SATURATION_ADU:.0f} ADU ceiling")

    # --- relative tests -----------------------------------------------------
    usable = [i for i, s in enumerate(small) if s is not None]

    # The reference for every RELATIVE test comes only from frames that already
    # passed the absolute ones. A folder holding a few real flats plus a pile of
    # junk would otherwise let the junk define "normal": the per-pixel median
    # consensus would be built from rubbish and the good frames would be the
    # ones flagged. Measured on a real 30-frame folder spanning 102-3800 ADU,
    # this is the difference between rejecting all 30 and rejecting the right 20.
    clean = [i for i in usable if not records[i]['reasons']]
    if len(clean) < 3:
        clean = usable                       # nothing better to compare against

    def _norm(i):
        s = small[i]
        m = np.median(s)
        return s / m if m > 0 else np.full_like(s, np.nan)

    spikes_clean = np.array([records[i]['spikes'] for i in clean], dtype='float64')
    if len(clean) >= 3 and spikes_clean.size:
        ref = max(float(np.median(spikes_clean)), 5.0)   # floor: ignore single digits
        for i in usable:
            if records[i]['spikes'] > SPIKE_RATIO * ref:
                records[i]['reasons'].append(
                    f"{records[i]['spikes']} bright spikes against a set median of "
                    f"{ref:.0f} -- stars or a satellite trail")

    if len(clean) >= 3:
        consensus = np.median(np.stack([_norm(i) for i in clean]), axis=0)

        sig, blk = {}, {}
        for i in usable:                     # measure EVERY frame, judge below
            d = _norm(i) - consensus
            sig[i] = _robust_sigma(d)
            blk[i] = _max_block_deviation(d)
            records[i]['consensus_sigma'] = sig[i]
            records[i]['block_deviation'] = blk[i]

        ref_sig = float(np.median([sig[i] for i in clean]))
        if ref_sig > 0:
            for i in usable:
                if sig[i] > CONSENSUS_RATIO * ref_sig:
                    records[i]['reasons'].append(
                        f"differs from the other flats by {sig[i] / ref_sig:.1f}x the "
                        f"usual scatter -- the illumination pattern does not match")

        # Localised test. The floor matters: without it a set of near-identical
        # frames would make the reference ~0 and reject on pure noise.
        ref_blk = max(float(np.median([blk[i] for i in clean])), MIN_BLOCK_DEVIATION)
        for i in usable:
            if blk[i] > BLOCK_RATIO * ref_blk:
                records[i]['reasons'].append(
                    f"part of the frame is {100 * blk[i]:.2f}% off the level the "
                    f"other flats agree on -- a sensor glitch, shadow or "
                    f"obstruction over one region")

    for rec in records:
        rec['ok'] = not rec['reasons']

    if progress_callback:
        progress_callback(1.0, "Screening complete")

    n_bad = sum(1 for r in records if not r['ok'])
    thresholds = {
        'linearity_limit': linearity_limit,
        'min_level': min_level,
        'saturation_adu': SATURATION_ADU,
        'max_saturated_fraction': MAX_SATURATED_FRACTION,
        'spike_ratio': SPIKE_RATIO,
        'consensus_ratio': CONSENSUS_RATIO,
        'block_ratio': BLOCK_RATIO,
        'min_block_deviation': MIN_BLOCK_DEVIATION,
    }
    if len(usable) < 3:
        summary = (f"{n_bad} of {n} frame(s) failed. Only {len(usable)} frame(s) "
                   f"could be compared, so the spike and consensus tests were "
                   f"skipped -- they need at least 3.")
    else:
        summary = f"{n_bad} of {n} frame(s) failed screening."
    return {'frames': records, 'thresholds': thresholds, 'summary': summary}
