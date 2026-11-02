"""
Photon-transfer measurement of camera conversion gain (e-/ADU) and read noise.

Method (standard flat/bias pair technique, e.g. Howell, Handbook of CCD
Astronomy, ch. 4; Janesick, Photon Transfer):

    gain [e-/ADU] = (mean(F1) + mean(F2) - mean(B1) - mean(B2))
                    / (var(F1 - F2) - var(B1 - B2))

    read_noise [ADU] = std(B1 - B2) / sqrt(2)
    read_noise [e-]  = read_noise_ADU * gain

Requirements for a good measurement:
  - F1, F2: two flat frames at IDENTICAL illumination, ~30-70% of saturation,
    same camera mode (High / Low / Merge), same exposure time.
  - B1, B2: two bias (zero-exposure) frames in the SAME camera mode.
  - Statistics are computed on a central region to avoid vignetting edges.
  - IMPORTANT: use frames at the SAME BINNING as your science frames.
    Binning changes the effective conversion gain: the KL4040's 2x2 output is
    k x (sum of four pixels), k = 0.40 (High) / 0.47 (Low), which multiplies
    e-/ADU by about 2.5. A gain measured on 1x1 frames does not apply to 2x2
    science frames.

Usage:
    python tools/measure_gain.py flat1.fits flat2.fits bias1.fits bias2.fits
    python tools/measure_gain.py f1 f2 b1 b2 --box 1000   (central box size)

Run once per camera mode and put the results into core/camera_config.py.
"""
import argparse
import sys

import numpy as np
from astropy.io import fits


def load(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            for h in hdul:
                if getattr(h, 'data', None) is not None:
                    data = h.data
                    break
        if data is None:
            raise ValueError(f"No image data in {path}")
        return np.squeeze(data).astype('float64')


def central_box(img, box):
    ny, nx = img.shape
    half = min(box, ny, nx) // 2
    cy, cx = ny // 2, nx // 2
    return img[cy - half:cy + half, cx - half:cx + half]


def main():
    ap = argparse.ArgumentParser(description="Photon-transfer gain measurement")
    ap.add_argument('flat1')
    ap.add_argument('flat2')
    ap.add_argument('bias1')
    ap.add_argument('bias2')
    ap.add_argument('--box', type=int, default=1000,
                    help="Side length (px) of the central analysis box (default 1000)")
    args = ap.parse_args()

    f1, f2 = load(args.flat1), load(args.flat2)
    b1, b2 = load(args.bias1), load(args.bias2)

    if not (f1.shape == f2.shape == b1.shape == b2.shape):
        sys.exit(f"Frame shapes differ: {f1.shape}, {f2.shape}, {b1.shape}, {b2.shape}")

    f1c, f2c = central_box(f1, args.box), central_box(f2, args.box)
    b1c, b2c = central_box(b1, args.box), central_box(b2, args.box)

    mean_f1, mean_f2 = np.mean(f1c), np.mean(f2c)
    mean_b1, mean_b2 = np.mean(b1c), np.mean(b2c)

    # Flat levels should match within a few percent, or the variance of the
    # difference picks up an illumination-drift term and biases the gain.
    if mean_f2 > 0 and abs(mean_f1 - mean_f2) / mean_f2 > 0.05:
        print(f"WARNING: flat levels differ by "
              f"{100 * abs(mean_f1 - mean_f2) / mean_f2:.1f}% -- "
              f"gain estimate may be biased. Use flats at identical illumination.")

    var_fdiff = np.var(f1c - f2c)
    var_bdiff = np.var(b1c - b2c)

    signal = mean_f1 + mean_f2 - mean_b1 - mean_b2
    denom = var_fdiff - var_bdiff
    if denom <= 0:
        sys.exit("var(F1-F2) <= var(B1-B2): flats carry no measurable shot noise. "
                 "Use brighter flats (~half saturation).")

    gain = signal / denom
    rn_adu = np.std(b1c - b2c) / np.sqrt(2.0)
    rn_e = rn_adu * gain

    print(f"Analysis box              : {f1c.shape[1]} x {f1c.shape[0]} px (central)")
    print(f"Flat means (bias-subtr.)  : {mean_f1 - mean_b1:.1f}, {mean_f2 - mean_b2:.1f} ADU")
    print(f"var(F1-F2)                : {var_fdiff:.2f} ADU^2")
    print(f"var(B1-B2)                : {var_bdiff:.2f} ADU^2")
    print(f"----------------------------------------")
    print(f"CONVERSION GAIN           : {gain:.4f} e-/ADU")
    print(f"READ NOISE                : {rn_adu:.3f} ADU  =  {rn_e:.3f} e-")
    print(f"----------------------------------------")
    print(f"Enter these into core/camera_config.py for the camera mode these")
    print(f"frames were saved in (gain=..., read_noise=<e- value>).")


if __name__ == '__main__':
    main()
