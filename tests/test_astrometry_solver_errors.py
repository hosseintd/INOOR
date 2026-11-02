"""
What the plate solver tells the operator when it cannot even start: no network, wrong API key.
The GUI shows result['error'] under "Astrometry.net could not solve the image". No network here.
"""
import numpy as np
import requests
from astropy.io import fits

import core.astrometry.astrometry_solver as sm


def _solver(tmp_path):
    path = tmp_path / 'frame.fits'
    fits.writeto(path, np.zeros((16, 16), np.float32))
    return sm.AstrometrySolver(api_key='some-key', timeout=30), str(path)


def test_no_network_is_named_as_such(tmp_path, monkeypatch):
    def refuse(*a, **kw):
        raise requests.exceptions.ConnectionError("HTTPConnectionPool(host='nova.astrometry.net', "
                                                  "port=80): Max retries exceeded")
    monkeypatch.setattr(sm.requests, 'post', refuse)
    s, path = _solver(tmp_path)
    res = s.solve_from_file(path, downsample_factor=1)
    assert res['solved_fits'] is None
    assert res['error'].startswith('Could not reach nova.astrometry.net. Check the internet connection.')


class _Answer:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_a_wrong_key_says_so(tmp_path, monkeypatch):
    # nova.astrometry.net's real answer to an unknown key (checked 2026-10-19)
    monkeypatch.setattr(sm.requests, 'post',
                        lambda *a, **kw: _Answer({'status': 'error', 'errormessage': 'bad apikey'}))
    s, path = _solver(tmp_path)
    res = s.solve_from_file(path, downsample_factor=1)
    assert res['solved_fits'] is None
    assert 'bad apikey' in res['error'] and 'Check the API key' in res['error']


def _server(job_status):
    """Login and upload succeed; the job then ends with job_status."""
    def post(url, **kw):
        if url.endswith('/api/login'):
            return _Answer({'status': 'success', 'session': 'abc'})
        return _Answer({'status': 'success', 'subid': 7})

    def get(url, **kw):
        if '/api/submissions/' in url:
            return _Answer({'jobs': [11]})
        return _Answer({'status': job_status})
    return post, get


def test_a_failed_job_message_reads_as_two_sentences(tmp_path, monkeypatch):
    post, get = _server('failure')
    monkeypatch.setattr(sm.requests, 'post', post)
    monkeypatch.setattr(sm.requests, 'get', get)
    s, path = _solver(tmp_path)
    res = s.solve_from_file(path, downsample_factor=1)
    assert res['solved_fits'] is None
    assert 'may be wrong. Try to decrease Downsample Factor' in res['error']
