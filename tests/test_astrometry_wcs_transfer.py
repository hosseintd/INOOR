"""
Guards how the plate solution from astrometry.net reaches INOOR's "solved" FITS file.

astrometry.net returns CTYPE = '...-TAN-SIP' plus a SIP polynomial (A_*, B_*, AP_*, BP_*)
for the optical distortion. The solved file used to copy CTYPE but none of the SIP keywords:
on PSR J2215 frame _High_86_ (0.0484"/px, 3.3' field) that moved positions by 0.062" rms
and 0.22" (4.6 px) at the frame edge. These tests use a synthetic SIP header, no network.
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points

from core.astrometry.astrometry_solver import (AstrometrySolver, _make_downsampled_header,
                                               _rebin_block_average, _scale_solved_header_to_fullres)

NPIX = 4096   # KL4040 frame; the WCS is evaluated on this grid, no image this size is made


def _sip_header(crpix1, crpix2, scale_deg, order=3, amp=3.0):
    """A TAN-SIP header like astrometry.net's, with distortion of a few px at the frame edge."""
    h = fits.Header()
    h['WCSAXES'] = 2
    h['CTYPE1'] = 'RA---TAN-SIP'
    h['CTYPE2'] = 'DEC--TAN-SIP'
    h['EQUINOX'] = 2000.0
    h['LONPOLE'] = 180.0
    h['LATPOLE'] = 0.0
    h['CRVAL1'] = 333.903663339
    h['CRVAL2'] = 51.5928033762
    h['CRPIX1'] = crpix1
    h['CRPIX2'] = crpix2
    h['CUNIT1'] = 'deg'
    h['CUNIT2'] = 'deg'
    h['CD1_1'] = -1.5e-3 * scale_deg
    h['CD1_2'] = -scale_deg
    h['CD2_1'] = scale_deg
    h['CD2_2'] = -1.9e-3 * scale_deg
    rng = np.random.default_rng(86)
    for name in ('A', 'B', 'AP', 'BP'):
        h[f'{name}_ORDER'] = order
    for p in range(order + 1):
        for q in range(order + 1 - p):
            if p + q < 2:
                continue
            # each term is ~amp px at u, v ~ 2000 px (frame 86: 4.6 px at the edge)
            a, b = rng.normal(0.0, amp, 2) / 2000.0 ** (p + q)
            h[f'A_{p}_{q}'], h[f'B_{p}_{q}'] = a, b
            h[f'AP_{p}_{q}'], h[f'BP_{p}_{q}'] = -a, -b
    return h


def _sip_keys(h):
    return {k for k in h if k.split('_')[0] in ('A', 'B', 'AP', 'BP') and '_' in k}


def _grid(n=65):
    g = np.linspace(0.0, NPIX - 1.0, n)     # 0-based pixel centres, frame edges included
    x, y = np.meshgrid(g, g)
    return x.ravel(), y.ravel()


def _sky_offset_px(w_a, xa, ya, w_b, xb, yb):
    """Angular distance between w_a(xa, ya) and w_b(xb, yb), in pixels of w_a."""
    ra_a, de_a = w_a.all_pix2world(xa, ya, 0)
    ra_b, de_b = w_b.all_pix2world(xb, yb, 0)
    sep_deg = np.hypot((ra_b - ra_a) * np.cos(np.deg2rad(de_a)), de_b - de_a)
    return sep_deg / np.sqrt(abs(np.linalg.det(w_a.wcs.cd)))


def test_apply_wcs_keeps_sip_and_the_frame_header(tmp_path):
    ref = _sip_header(2204.81, 1051.80, 0.0484 / 3600.0)
    wcs_path = tmp_path / 'wcs.fits'
    fits.PrimaryHDU(header=ref).writeto(wcs_path)

    data = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
    orig = fits.Header()
    orig['EXPTIME'] = 9399
    orig['GAIN'] = 16.5
    orig['DATE-OBS'] = '2024-09-09T23:34:47'
    orig['A_5_0'] = 1.0e-3      # stale SIP term from an older solve: must not survive
    img_path = tmp_path / 'frame.fits'
    fits.PrimaryHDU(data=data, header=orig).writeto(img_path)

    out_path = tmp_path / 'solved.fits'
    assert AstrometrySolver(api_key='unused')._apply_wcs_to_image(
        str(img_path), str(wcs_path), str(out_path)) is True

    with fits.open(out_path) as hd:
        out = hd[0].header
        assert np.array_equal(hd[0].data, data)
    for k in ('EXPTIME', 'GAIN', 'DATE-OBS'):
        assert out[k] == orig[k]
    assert _sip_keys(out) == _sip_keys(ref)

    w_ref, w_out = WCS(ref), WCS(out)
    x, y = _grid()
    # same pixel -> same sky, over the whole frame
    assert _sky_offset_px(w_ref, x, y, w_out, x, y).max() < 1e-6
    # pixel -> sky -> pixel through the solved file's own WCS
    ra, dec = w_out.all_pix2world(x, y, 0)
    xb, yb = w_out.all_world2pix(ra, dec, 0, tolerance=1e-12, maxiter=200)
    assert np.hypot(xb - x, yb - y).max() < 1e-6


@pytest.mark.parametrize('factor', [2, 3, 4])
def test_fullres_scaling_carries_sip(factor):
    """The full-res header must describe the same sky as the down-sampled solution, under the
    pixel mapping its own CRPIX pair implies: X - CRPIX_full = f * (x - CRPIX_down)."""
    down = _sip_header(2204.81 / factor, 1051.80 / factor, factor * 0.0484 / 3600.0)
    full = _scale_solved_header_to_fullres(down, factor, NPIX, NPIX)
    w_down, w_full = WCS(down), WCS(full)

    X, Y = _grid()
    x = (X + 1.0 - full['CRPIX1']) / factor + down['CRPIX1'] - 1.0
    y = (Y + 1.0 - full['CRPIX2']) / factor + down['CRPIX2'] - 1.0
    assert _sky_offset_px(w_full, X, Y, w_down, x, y).max() < 1e-6

    # inverse polynomial (AP, BP). With origin=1 astropy's sip_foc2pix takes offsets U from CRPIX
    # and returns CRPIX + U + AP(U), so U + AP(U) in full-res px must be f times the down value.
    crpix_full = np.array([full['CRPIX1'], full['CRPIX2']])
    crpix_down = np.array([down['CRPIX1'], down['CRPIX2']])
    U_full = np.column_stack([X, Y]) + 1.0 - crpix_full
    back_full = w_full.sip_foc2pix(U_full, 1) - crpix_full
    back_down = w_down.sip_foc2pix(U_full / factor, 1) - crpix_down
    assert np.abs(back_full - factor * back_down).max() < 1e-6


# ---- downsample path: INOOR solves a block-averaged copy and scales the answer back ----------
# Down pixel p is the mean of an f x f block, so (0-based) its centre is full-res X = f*p + (f-1)/2.
# The old CRPIX rule f*(p-1)+1 put every position (f-1)/2 full px off in both directions.

SMALL = 240        # divisible by 2, 3, 4


def _star_field(w_full, sig=4.0):
    """Gaussian stars at known sub-pixel positions in a SMALL x SMALL image, ~60 px apart so
    that no star reaches a neighbour's centroid window."""
    rng = np.random.default_rng(7)
    g = np.linspace(30.0, SMALL - 31.0, 4)
    xs, ys = (a.ravel() + rng.uniform(-0.5, 0.5, a.size) for a in np.meshgrid(g, g))
    yy, xx = np.mgrid[0:SMALL, 0:SMALL]
    img = np.zeros((SMALL, SMALL))
    for x0, y0 in zip(xs, ys):
        img += np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sig ** 2))
    ra, dec = w_full.all_pix2world(xs, ys, 0)
    return img, xs, ys, ra, dec


def _down_centroids(img, xs, ys, factor, sig=4.0):
    """First-moment centroids (0-based down pixels) in INOOR's own block-averaged image."""
    ds = _rebin_block_average(img, factor)
    half = int(np.ceil(6.0 * sig / factor)) + 1
    out = []
    for x0, y0 in zip(xs, ys):
        xi, yi = int(round(x0 / factor)), int(round(y0 / factor))
        sub = ds[yi - half:yi + half + 1, xi - half:xi + half + 1]
        gy, gx = np.mgrid[yi - half:yi + half + 1, xi - half:xi + half + 1]
        out.append(((sub * gx).sum() / sub.sum(), (sub * gy).sum() / sub.sum()))
    return np.array(out).T


@pytest.mark.parametrize('factor', [2, 3, 4])
def test_downsampled_header_matches_block_average(factor):
    # forward, with SIP: where the down header puts each star vs where the star is in the image
    full = _sip_header(118.3, 124.6, 0.0484 / 3600.0, amp=3.0 * (SMALL / 4096.0))
    full['NAXIS'], full['NAXIS1'], full['NAXIS2'] = 2, SMALL, SMALL
    img, xs, ys, ra, dec = _star_field(WCS(full))
    cx, cy = _down_centroids(img, xs, ys, factor)
    down = _make_downsampled_header(full, factor, SMALL // factor, SMALL // factor)
    px, py = WCS(down).all_world2pix(ra, dec, 0, tolerance=1e-12, maxiter=200)
    assert np.hypot(px - cx, py - cy).max() * factor < 1e-4

    # inverse: 'solve' the down image from the measured centroids alone (as astrometry.net
    # solves what it is sent), scale back, compare with the true full-res positions
    tan = full.copy()
    for k in [k for k in tan if k.split('_')[0] in ('A', 'B', 'AP', 'BP') and '_' in k]:
        del tan[k]
    tan['CTYPE1'], tan['CTYPE2'] = 'RA---TAN', 'DEC--TAN'
    img, xs, ys, ra, dec = _star_field(WCS(tan))
    cx, cy = _down_centroids(img, xs, ys, factor)
    solved = fit_wcs_from_points((cx, cy), SkyCoord(ra, dec, unit='deg'), projection='TAN')
    back = _scale_solved_header_to_fullres(solved.to_header(), factor, SMALL, SMALL)
    bx, by = WCS(back).all_world2pix(ra, dec, 0)
    assert np.hypot(bx - xs, by - ys).max() < 1e-4


@pytest.mark.parametrize('factor', [2, 3, 4])
def test_downsample_mapping_with_sip_over_full_frame(factor):
    """The SIP rescaling A_pq * f**(1-p-q), both directions, to < 1e-6 px over a 4096 frame."""
    X, Y = _grid()
    x, y = (X - (factor - 1) / 2.0) / factor, (Y - (factor - 1) / 2.0) / factor

    full = _sip_header(2204.81, 1051.80, 0.0484 / 3600.0)
    down = _make_downsampled_header(full, factor, NPIX // factor, NPIX // factor)
    assert _sky_offset_px(WCS(full), X, Y, WCS(down), x, y).max() < 1e-6

    solved = _sip_header(2204.81 / factor, 1051.80 / factor, factor * 0.0484 / 3600.0)
    back = _scale_solved_header_to_fullres(solved, factor, NPIX, NPIX)
    assert _sky_offset_px(WCS(back), X, Y, WCS(solved), x, y).max() < 1e-6

    # down and back again is the identity
    again = _scale_solved_header_to_fullres(down, factor, NPIX, NPIX)
    assert _sky_offset_px(WCS(full), X, Y, WCS(again), X, Y).max() < 1e-6
