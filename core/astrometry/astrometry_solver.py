# core/astrometry/astrometry_solver.py
import os
import re
import sys
import time
import json
import tempfile
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord

import requests

# Keywords of a plate solution that are copied onto the original image (structural ones such as
# NAXIS/BITPIX stay the image's own).
_WCS_KEYWORDS = (
    'RADESYS', 'EQUINOX', 'WCSAXES', 'LONPOLE', 'LATPOLE',
    'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
    'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
    'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
    'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2',
    'PV1_1', 'PV1_2', 'PV1_3', 'PV1_4', 'PV1_5',
    'PV2_1', 'PV2_2', 'PV2_3', 'PV2_4', 'PV2_5',
)
# SIP distortion polynomial: forward (A, B) and inverse (AP, BP). It must travel with
# CTYPE '...-TAN-SIP'. Dropping it (28 keywords on PSR J2215 frame _High_86_, 0.0484"/px)
# moved positions by 0.062" rms and 0.22" (4.6 px) at the frame edge.
_SIP_ORDER_KEY = re.compile(r'^(A|B|AP|BP)_ORDER$')
_SIP_TERM_KEY = re.compile(r'^(A|B|AP|BP)_(\d+)_(\d+)$')


def _is_sip_keyword(key):
    return bool(_SIP_ORDER_KEY.match(key) or _SIP_TERM_KEY.match(key))


def _copy_wcs_keywords(src_hdr, dst_hdr):
    """Copy the plate solution in src_hdr onto dst_hdr in place; other keywords stay untouched.
    SIP keywords already in dst_hdr (an older solve) are removed first, so a new solution of
    lower order, or one without SIP, cannot inherit stale terms."""
    for key in [k for k in dst_hdr.keys() if _is_sip_keyword(k)]:
        del dst_hdr[key]
    for key in _WCS_KEYWORDS:
        if key in src_hdr:
            dst_hdr[key] = (src_hdr[key], src_hdr.comments[key])
    for key in src_hdr.keys():
        if _is_sip_keyword(key):
            dst_hdr[key] = (src_hdr[key], src_hdr.comments[key])


def _rebin_block_average(data, factor):
    if factor == 1:
        return data.copy()
    ny, nx = data.shape
    ny_crop = (ny // factor) * factor
    nx_crop = (nx // factor) * factor
    data_cropped = data[:ny_crop, :nx_crop]
    reshaped = data_cropped.reshape((ny_crop//factor, factor, nx_crop//factor, factor))
    rebinned = reshaped.mean(axis=(1, 3))
    return rebinned

def _scale_sip_terms(hdr, s):
    """Rescale the SIP polynomial in place for pixels s times larger in count (s = f going to
    full resolution, 1/f going down). SIP terms are pixel offsets polynomial in pixel offsets
    from CRPIX, so s * A_pq * (u/s)^p * (v/s)^q gives A_pq -> A_pq * s**(1-p-q); the inverse
    (AP, BP) maps pixels to pixels and scales alike."""
    for key in list(hdr.keys()):
        m = _SIP_TERM_KEY.match(key)
        if m:
            p, q = int(m.group(2)), int(m.group(3))
            hdr[key] = float(hdr[key]) * s ** (1 - p - q)


# Pixel mapping of _rebin_block_average (crop from pixel 1, mean of f x f blocks): down pixel p
# (1-based FITS) is the mean of full pixels f*(p-1)+1 ... f*p, so its centre is at full-res
#     X = f*p - (f-1)/2,        inversely  p = (X + (f-1)/2) / f.
# The old f*(p-1)+1 put every position (f-1)/2 full px off (0.5 px at f = 2, 1.5 px at f = 4),
# measured with block-averaged synthetic stars in both directions.

def _make_downsampled_header(orig_header, factor, new_nx, new_ny):
    hdr = orig_header.copy()
    hdr['NAXIS1'] = new_nx
    hdr['NAXIS2'] = new_ny
    for pix in (1, 2):
        key = f'CRPIX{pix}'
        if key in hdr:
            hdr[key] = (float(hdr[key]) + (factor - 1) / 2.0) / factor
    if any(k in hdr for k in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')):
        for k in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'):
            if k in hdr:
                try:
                    hdr[k] = float(hdr[k]) * float(factor)
                except Exception:
                    pass
    else:
        for k in ('CDELT1', 'CDELT2'):
            if k in hdr:
                try:
                    hdr[k] = float(hdr[k]) * float(factor)
                except Exception:
                    pass
    _scale_sip_terms(hdr, 1.0 / factor)
    return hdr

def _scale_solved_header_to_fullres(solved_hdr, factor, orig_nx, orig_ny):
    new_hdr = solved_hdr.copy()
    new_hdr['NAXIS1'] = orig_nx
    new_hdr['NAXIS2'] = orig_ny
    for pix in (1, 2):
        key = f'CRPIX{pix}'
        if key in new_hdr:
            new_hdr[key] = factor * float(new_hdr[key]) - (factor - 1) / 2.0
    if any(k in new_hdr for k in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')):
        for k in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'):
            if k in new_hdr:
                try:
                    new_hdr[k] = float(new_hdr[k]) / float(factor)
                except Exception:
                    pass
    else:
        for k in ('CDELT1', 'CDELT2'):
            if k in new_hdr:
                try:
                    new_hdr[k] = float(new_hdr[k]) / float(factor)
                except Exception:
                    pass
    _scale_sip_terms(new_hdr, float(factor))
    return new_hdr

class AstrometrySolver:
    # Resolve project root from this file's location: core/astrometry/ -> project root
    _PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

    def __init__(self, api_key=None, timeout=900, poll_interval=3):
        self.api_key = api_key or os.environ.get('ASTROMETRY_API_KEY', '')
        self.timeout = timeout
        self.poll_interval = poll_interval

    @staticmethod
    def _get_output_base():
        """Return absolute path to the Output/Astrometry directory.
        When frozen (PyInstaller exe), uses the exe's directory.
        When running as script, uses the project root."""
        if getattr(sys, 'frozen', False):
            base = os.path.dirname(sys.executable)
        else:
            base = AstrometrySolver._PROJECT_ROOT
        return os.path.join(base, 'Output', 'Astrometry')

    def _ensure_job_folder(self, job_id, submission_id=None):
        base_dir = self._get_output_base()
        job_subdir = f"job_{job_id}" if job_id else f"sub_{submission_id}"
        full_path = os.path.join(base_dir, job_subdir)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def _create_downsampled_fits(self, orig_path, factor):
        with fits.open(orig_path, ignore_missing_end=True, ignore_missing_simple=True) as hd:
            hdu_img = hd[0]
            data = hdu_img.data
            hdr = hdu_img.header
            if data is None:
                for h in hd:
                    if getattr(h, 'data', None) is not None:
                        data = h.data
                        hdr = h.header
                        break
            if data is None:
                raise RuntimeError('No image data found in FITS file.')
            if data.ndim != 2:
                data = data.squeeze()
                if data.ndim != 2:
                    raise RuntimeError('Image data is not 2D.')
            ny, nx = data.shape
            ny_crop = (ny // factor) * factor
            nx_crop = (nx // factor) * factor
            if ny_crop == 0 or nx_crop == 0:
                raise ValueError('Downsample factor too large for image size.')
            data_crop = data[:ny_crop, :nx_crop].astype(float)
            ds = _rebin_block_average(data_crop, factor)
            hdr_down = _make_downsampled_header(hdr, factor, ds.shape[1], ds.shape[0])
            
            # Using current project's temp approach
            tmp_fd, tmp_path = tempfile.mkstemp(prefix='downsampled_', suffix='.fits')
            os.close(tmp_fd)
            try:
                fits.PrimaryHDU(data=ds.astype(np.float32), header=hdr_down).writeto(tmp_path, overwrite=True)
            except Exception:
                os.unlink(tmp_path)
                raise
            return tmp_path, (ny, nx)

    def _get_job_for_submission(self, submission_id, timeout=30):
        """Wait until the submission has a job_id, then return it."""
        sub_url = f'http://nova.astrometry.net/api/submissions/{submission_id}'
        t0 = time.time()
        while True:
            r = requests.get(sub_url, timeout=30)
            r.raise_for_status()
            data = r.json()
            jobs = data.get('jobs', [])
            if jobs and len(jobs) > 0 and jobs[0] is not None:
                return jobs[0]
            if time.time() - t0 > timeout:
                return None
            time.sleep(self.poll_interval)

    def _wait_for_job_result(self, job_id, timeout=900):
        """Poll job status until it completes. Returns 'success', 'failure', or None on timeout."""
        job_url = f'http://nova.astrometry.net/api/jobs/{job_id}'
        t0 = time.time()
        while True:
            try:
                r = requests.get(job_url, timeout=30)
                r.raise_for_status()
                data = r.json()
                status = data.get('status', '')
                if status == 'success':
                    return 'success'
                if status == 'failure':
                    return 'failure'
            except Exception as e:
                print(f'Job status poll error: {e}')
            if time.time() - t0 > timeout:
                return None
            time.sleep(self.poll_interval)

    def _is_valid_fits_content(self, content):
        """Check if the content is a valid FITS file, not an HTML error page."""
        if not content:
            return False
        # Check for HTML signatures
        if content.startswith(b'<!DOCTYPE') or content.startswith(b'<html') or content.startswith(b'<HTML'):
            return False
        # Check for FITS magic number
        if content.startswith(b'SIMPLE  '):
            return True
        return False

    def _download_result_file(self, job_id, endpoint_name, out_filename):
        url = f'http://nova.astrometry.net/{endpoint_name}/{job_id}'
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            
            # Validate content is FITS, not HTML error
            if not self._is_valid_fits_content(r.content):
                if r.content.startswith(b'<!DOCTYPE') or r.content.startswith(b'<html'):
                    print(f'Download {endpoint_name} returned HTML (likely human verification needed)')
                    return None
                print(f'Downloaded {endpoint_name} does not appear to be valid FITS')
                return None
            
            with open(out_filename, 'wb') as fh:
                fh.write(r.content)
            return os.path.abspath(out_filename)
        except Exception as e:
            print(f'Failed to download {endpoint_name} for job {job_id}:', e)
            return None

    def _apply_wcs_to_image(self, original_fits_path, wcs_fits_path, output_path):
        """Extract WCS header from wcs.fits and apply to original image data.
        Returns True if successful, False otherwise."""
        try:
            # Read original image data
            with fits.open(original_fits_path, ignore_missing_end=True, ignore_missing_simple=True) as orig_hdul:
                orig_data = orig_hdul[0].data
                orig_header = orig_hdul[0].header.copy()
                if orig_data is None:
                    for h in orig_hdul:
                        if getattr(h, 'data', None) is not None:
                            orig_data = h.data
                            orig_header = h.header.copy()
                            break
                
                if orig_data is None:
                    print('No image data found in original FITS')
                    return False
            
            with fits.open(wcs_fits_path, ignore_missing_end=True, ignore_missing_simple=True) as wcs_hdul:
                wcs_header = wcs_hdul[0].header

                # Create new HDU with original data
                new_hdu = fits.PrimaryHDU(data=orig_data, header=orig_header)

                # Copy only WCS-related keywords (SIP included) to avoid header conflicts
                _copy_wcs_keywords(wcs_header, new_hdu.header)

                new_hdu.writeto(output_path, overwrite=True)
                return True
        except Exception as e:
            print(f'Failed to apply WCS to image: {e}')
            return False

    def solve_from_file(self, fits_path, scale_lower_arcmin=2.0, scale_upper_arcmin=10.0, solve_timeout=None, downsample_factor: int = 1, ra=None, dec=None, search_radius=None):
        solve_timeout = solve_timeout or self.timeout

        if ra is not None and dec is not None:
            if search_radius is not None:
                radius_deg = float(search_radius) / 60.0
                print(f'Plate solving with RA/Dec hint: ({ra:.4f}, {dec:.4f}) radius: {search_radius} arcmin ({radius_deg:.2f}°)')
            else:
                print(f'Plate solving with RA/Dec hint: ({ra:.4f}, {dec:.4f})')
        else:
            print('Plate solving without RA/Dec hint')

        # The image itself is uploaded and astrometry.net finds the stars. astroquery's
        # solve_from_image used to be tried first wherever it imported: with photutils installed
        # it detects stars locally (DAOStarFinder, fwhm=3, which splits saturated INO340 stars
        # into several sources) and sends only that list, and its status polling gives up on a
        # dropped connection (seen behind a local proxy). PSR High_86, f=4: 106 vs 42 mas rms
        # against Gaia, 435 s vs 35-78 s.
        if not self.api_key:
            raise RuntimeError('No astrometry.net API key given.')

        ds_path = None
        orig_dims = None
        factor = int(downsample_factor) if downsample_factor else 1
        if factor > 1:
            ds_path, orig_dims = self._create_downsampled_fits(fits_path, factor)
            submit_path = ds_path
        else:
            submit_path = fits_path

        try:
            login_url = 'http://nova.astrometry.net/api/login'
            # Use data=... not json=... specifically for some versions, but here we keep existing
            r = requests.post(login_url, data={'request-json': json.dumps({'apikey': self.api_key})}, timeout=30)
            r.raise_for_status()
            resp = r.json()
            session = resp.get('session')
            if not session:
                # A wrong key is answered with {"status": "error", "errormessage": "bad apikey"}.
                raise RuntimeError(f"Login to astrometry.net failed "
                                   f"({resp.get('errormessage', 'no session returned')}). "
                                   f"Check the API key.")

            upload_url = 'http://nova.astrometry.net/api/upload'
            request_json = {'session': session, 'publicly_visible': 'n',
                        'scale_units': 'arcminwidth', 'scale_type': 'ul',
                        'scale_lower': float(scale_lower_arcmin), 'scale_upper': float(scale_upper_arcmin)}
            
            # Add RA/Dec hint if provided
            if ra is not None and dec is not None:
                request_json['center_ra'] = float(ra)
                request_json['center_dec'] = float(dec)
                if search_radius is not None:
                    request_json['radius'] = float(search_radius) / 60.0
            
            with open(submit_path, 'rb') as fh:
                files = {'file': (os.path.basename(submit_path), fh)}
                data = {'request-json': json.dumps(request_json)}
                r2 = requests.post(upload_url, files=files, data=data, timeout=60)
            r2.raise_for_status()
            
            sub = r2.json()
            submission_id = sub.get('subid') or sub.get('submission_id') or sub.get('id')
            if not submission_id:
                raise RuntimeError('Upload did not return a submission id')

            job_id = self._get_job_for_submission(submission_id, timeout=solve_timeout)
            solved_fits = None
            rdls_path, axy_path = None, None
            last_error = None
            
            if job_id is not None:
                # Wait for the job to actually complete and check its status
                job_status = self._wait_for_job_result(job_id, timeout=solve_timeout)
                if job_status == 'failure':
                    return {'solved_fits': None, 'submission_id': submission_id, 'job_id': job_id,
                            'error': f'Astrometry.net job {job_id} failed to solve the image. '
                                     f'The field may have too few stars or the scale hint may be wrong. '
                                     f"Try to decrease Downsample Factor or increase the Scale Upper parameter."}
                elif job_status is None:
                    return {'solved_fits': None, 'submission_id': submission_id, 'job_id': job_id,
                            'error': f'Timed out waiting for job {job_id} to complete.'}
                
                job_folder = self._ensure_job_folder(job_id, submission_id)
                
                # Download WCS file and save it
                wcs_fits_path = os.path.join(job_folder, f'wcs_{job_id}.fits')
                wcs_url = f'http://nova.astrometry.net/wcs_file/{job_id}'
                try:
                    r_wcs = requests.get(wcs_url, timeout=60)
                    r_wcs.raise_for_status()
                    
                    # Validate content is FITS, not HTML error
                    if not self._is_valid_fits_content(r_wcs.content):
                        if r_wcs.content.startswith(b'<!DOCTYPE') or r_wcs.content.startswith(b'<html'):
                            print(f'WCS file download returned HTML (human verification needed)')
                            last_error = 'Astrometry.net requires human verification for this solve. WCS file unavailable.'
                        else:
                            print(f'WCS file does not appear to be valid FITS')
                            last_error = 'Failed to download valid WCS file from Astrometry.net'
                    else:
                        # Save the WCS file
                        with open(wcs_fits_path, 'wb') as of:
                            of.write(r_wcs.content)
                        
                        # Apply WCS to original image
                        if ds_path and orig_dims is not None:
                            # Scale WCS header to full resolution and apply to original image
                            with fits.open(wcs_fits_path, ignore_missing_end=True, ignore_missing_simple=True) as wcs_hdul:
                                wcs_header = wcs_hdul[0].header
                            ny_orig, nx_orig = orig_dims
                            
                            with fits.open(fits_path, ignore_missing_end=True, ignore_missing_simple=True) as orig_hdul:
                                orig_data = orig_hdul[0].data
                                orig_header = orig_hdul[0].header.copy()
                                if orig_data is None:
                                    for h in orig_hdul:
                                        if getattr(h, 'data', None) is not None:
                                            orig_data = h.data
                                            orig_header = h.header.copy()
                                            break
                                
                                # Create new HDU with original data
                                new_hdu = fits.PrimaryHDU(data=orig_data, header=orig_header)

                                # Scale WCS to full resolution
                                hdr_full = _scale_solved_header_to_fullres(wcs_header, factor, nx_orig, ny_orig)

                                # Copy only WCS keywords (SIP included) to avoid structural conflicts
                                _copy_wcs_keywords(hdr_full, new_hdu.header)

                                final_solved = os.path.join(job_folder, f"solved_fullres_{os.path.basename(fits_path)}")
                                new_hdu.writeto(final_solved, overwrite=True)
                                solved_fits = final_solved
                        else:
                            # No downsampling, apply WCS directly to original
                            solved_fits = os.path.join(job_folder, f"solved_{os.path.basename(submit_path)}")
                            self._apply_wcs_to_image(fits_path, wcs_fits_path, solved_fits)
                except Exception as e:
                    last_error = f"WCS file download exception: {e}"
                    print(last_error)
                
                rdls_path = self._download_result_file(job_id, 'rdls_file', os.path.join(job_folder, f'rdls_{job_id}.fits'))
                axy_path  = self._download_result_file(job_id, 'axy_file',  os.path.join(job_folder, f'axy_{job_id}.fits'))
            else:
                 last_error = "Timed out waiting for job_id or submission failed."

            return {'solved_fits': solved_fits, 'submission_id': submission_id, 'job_id': job_id,
                    'rdls_path': rdls_path, 'axy_path': axy_path, 'downsample_factor': factor, 'error': last_error}

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_e:
            # The raw text ("HTTPConnectionPool(...): Max retries exceeded ...") read as if
            # the image had failed to solve.
            return {'solved_fits': None,
                    'error': f'Could not reach nova.astrometry.net. Check the internet '
                             f'connection. ({type(net_e).__name__})'}
        except Exception as global_e:
             return {'solved_fits': None, 'error': str(global_e)}
        finally:
            # Every way out: a failed job, a timeout, no network, a wrong key.
            if ds_path and os.path.exists(ds_path):
                try:
                    os.unlink(ds_path)
                except OSError as e:
                    print(f'Could not delete the temporary file {ds_path}: {e}')
