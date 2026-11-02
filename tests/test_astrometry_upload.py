"""
The plate solver uploads the image and lets astrometry.net find the stars.

Where astroquery imported (any fresh install: keyring comes with it), its solve_from_image was
tried first, and with photutils installed it sent a locally detected source list instead
(DAOStarFinder, fwhm=3). No network here.
"""
import numpy as np
import requests
from astropy.io import fits

import core.astrometry.astrometry_solver as sm


class _Answer:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_the_frame_itself_is_uploaded(tmp_path, monkeypatch):
    # a few stars on noise, so a local star finder would have something to send
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:64, 0:64]
    img = rng.normal(100.0, 5.0, (64, 64))
    for x0, y0 in ((15, 20), (40, 45), (50, 12), (25, 52)):
        img += 2000.0 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / 4.0)
    path = tmp_path / 'frame.fits'
    fits.writeto(path, img.astype(np.float32))
    calls = []

    def post(url, **kw):
        calls.append((url, kw))
        if url.endswith('/api/login'):
            return _Answer({'status': 'success', 'session': 'abc'})
        raise requests.exceptions.ConnectionError('stop after the upload request')

    # astroquery talks through a requests.Session; nothing may go that way
    session_sends = []

    def send(self, request, **kw):
        session_sends.append(request.url)
        raise requests.exceptions.ConnectionError('no network in tests')

    monkeypatch.setattr(sm.requests, 'post', post)
    monkeypatch.setattr(requests.sessions.Session, 'send', send)
    sm.AstrometrySolver(api_key='some-key', timeout=30).solve_from_file(str(path), downsample_factor=1)

    assert session_sends == []
    assert [url for url, _ in calls] == ['http://nova.astrometry.net/api/login',
                                         'http://nova.astrometry.net/api/upload']
    assert calls[1][1]['files']['file'][0] == 'frame.fits'
