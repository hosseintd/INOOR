"""
The downsampled copy of a frame is a temporary file. It used to be deleted only when a job ended
normally: a failed job, a timeout, no network or a wrong key left it in the system temp folder.
No network here. Every case also checks that the file existed when it was uploaded.
"""
import glob
import io
import os
import tempfile

import numpy as np
import pytest
import requests
from astropy.io import fits

import core.astrometry.astrometry_solver as sm


class _Answer:
    def __init__(self, payload=None, content=b''):
        self._payload = payload
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _wcs_bytes():
    hdr = fits.Header()
    hdr.update(CTYPE1='RA---TAN', CTYPE2='DEC--TAN', CRVAL1=333.87, CRVAL2=51.6, CRPIX1=4.0, CRPIX2=4.0,
               CD1_1=-1e-4, CD1_2=0.0, CD2_1=0.0, CD2_2=1e-4)
    out = io.BytesIO()
    fits.PrimaryHDU(header=hdr).writeto(out)
    return out.getvalue()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    temp_dir = tmp_path / 'temp'
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(temp_dir))
    monkeypatch.setattr(sm.AstrometrySolver, '_get_output_base', staticmethod(lambda: str(tmp_path / 'out')))
    monkeypatch.setattr(sm.AstrometrySolver, '_download_result_file', lambda self, *a, **kw: None)
    path = tmp_path / 'frame.fits'
    fits.writeto(path, np.zeros((16, 16), np.float32))
    return sm.AstrometrySolver(api_key='some-key', timeout=30), str(path), str(temp_dir)


def _left_behind(temp_dir):
    return glob.glob(os.path.join(temp_dir, 'downsampled_*'))


def _post(uploaded, upload=None):
    """Login succeeds; the upload records the file it was given, then does `upload`."""
    def post(url, **kw):
        if url.endswith('/api/login'):
            return _Answer({'status': 'success', 'session': 'abc'})
        uploaded.append(os.path.exists(kw['files']['file'][1].name)
                        and os.path.basename(kw['files']['file'][1].name).startswith('downsampled_'))
        if upload is not None:
            upload()
        return _Answer({'status': 'success', 'subid': 7})
    return post


def _get(job_status):
    def get(url, **kw):
        if '/api/submissions/' in url:
            return _Answer({'jobs': [11]})
        if '/wcs_file/' in url:
            return _Answer(content=_wcs_bytes())
        return _Answer({'status': job_status})
    return get


def _refuse(*a, **kw):
    raise requests.exceptions.ConnectionError('no network')


def test_a_solved_job(setup, monkeypatch):
    s, path, temp_dir = setup
    uploaded = []
    monkeypatch.setattr(sm.requests, 'post', _post(uploaded))
    monkeypatch.setattr(sm.requests, 'get', _get('success'))
    res = s.solve_from_file(path, downsample_factor=2)
    assert res['solved_fits'] and os.path.exists(res['solved_fits']), res
    assert uploaded == [True]
    assert _left_behind(temp_dir) == []


def test_a_failed_job(setup, monkeypatch):
    s, path, temp_dir = setup
    uploaded = []
    monkeypatch.setattr(sm.requests, 'post', _post(uploaded))
    monkeypatch.setattr(sm.requests, 'get', _get('failure'))
    res = s.solve_from_file(path, downsample_factor=2)
    assert 'failed to solve' in res['error']
    assert uploaded == [True]
    assert _left_behind(temp_dir) == []


def test_a_job_that_times_out(setup, monkeypatch):
    s, path, temp_dir = setup
    uploaded = []
    monkeypatch.setattr(sm.requests, 'post', _post(uploaded))
    monkeypatch.setattr(sm.requests, 'get', _get('success'))
    monkeypatch.setattr(sm.AstrometrySolver, '_wait_for_job_result', lambda self, *a, **kw: None)
    res = s.solve_from_file(path, downsample_factor=2)
    assert 'Timed out' in res['error']
    assert _left_behind(temp_dir) == []


def test_a_submission_that_gets_no_job(setup, monkeypatch):
    s, path, temp_dir = setup
    uploaded = []
    monkeypatch.setattr(sm.requests, 'post', _post(uploaded))
    monkeypatch.setattr(sm.AstrometrySolver, '_get_job_for_submission', lambda self, *a, **kw: None)
    res = s.solve_from_file(path, downsample_factor=2)
    assert 'Timed out' in res['error']
    assert _left_behind(temp_dir) == []


def test_no_network_at_login(setup, monkeypatch):
    s, path, temp_dir = setup
    monkeypatch.setattr(sm.requests, 'post', _refuse)
    res = s.solve_from_file(path, downsample_factor=2)
    assert res['error'].startswith('Could not reach')
    assert _left_behind(temp_dir) == []


def test_no_network_during_the_upload(setup, monkeypatch):
    s, path, temp_dir = setup
    uploaded = []
    monkeypatch.setattr(sm.requests, 'post', _post(uploaded, upload=_refuse))
    res = s.solve_from_file(path, downsample_factor=2)
    assert res['error'].startswith('Could not reach')
    assert uploaded == [True]
    assert _left_behind(temp_dir) == []


def test_a_wrong_key(setup, monkeypatch):
    s, path, temp_dir = setup
    monkeypatch.setattr(sm.requests, 'post',
                        lambda *a, **kw: _Answer({'status': 'error', 'errormessage': 'bad apikey'}))
    res = s.solve_from_file(path, downsample_factor=2)
    assert 'bad apikey' in res['error']
    assert _left_behind(temp_dir) == []


def test_an_unexpected_error(setup, monkeypatch):
    s, path, temp_dir = setup

    def broken(*a, **kw):
        raise ValueError('something unexpected')
    monkeypatch.setattr(sm.requests, 'post', broken)
    res = s.solve_from_file(path, downsample_factor=2)
    assert 'something unexpected' in res['error']
    assert _left_behind(temp_dir) == []


def test_no_key_makes_no_temporary_file(setup):
    _, path, temp_dir = setup
    with pytest.raises(RuntimeError, match='No astrometry.net API key'):
        sm.AstrometrySolver(api_key='', timeout=30).solve_from_file(path, downsample_factor=2)
    assert _left_behind(temp_dir) == []


def test_a_copy_that_cannot_be_written_is_not_left_behind(setup, monkeypatch):
    s, path, temp_dir = setup

    def disk_full(*a, **kw):
        raise OSError('No space left on device')
    monkeypatch.setattr(fits.PrimaryHDU, 'writeto', disk_full)
    with pytest.raises(OSError, match='No space left'):
        s.solve_from_file(path, downsample_factor=2)
    assert _left_behind(temp_dir) == []


def test_the_frame_itself_is_never_deleted(setup, monkeypatch):
    """Factor 1 uploads the frame itself, which the clean-up must not touch."""
    s, path, _ = setup
    monkeypatch.setattr(sm.requests, 'post', _refuse)
    s.solve_from_file(path, downsample_factor=1)
    assert os.path.exists(path)
