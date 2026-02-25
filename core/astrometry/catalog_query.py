"""Pan-STARRS (PS1) query helpers using the MAST catalogs REST API (no astroquery dependency required)."""
import requests
import json
from astropy.table import Table
from astropy.io import ascii

BASEURL = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"

def checklegal(table, release):
    if release not in ('dr1','dr2'):
        raise ValueError('release must be dr1 or dr2')
    if release == 'dr1' and table not in ('mean','stack'):
        raise ValueError('for dr1 table must be mean or stack')
    if release == 'dr2' and table not in ('mean','stack','detection'):
        raise ValueError('for dr2 table must be mean, stack, or detection')

def ps1search(table='mean', release='dr1', format='csv', columns=None, verbose=False, **kw):
    data = kw.copy()
    if not data:
        raise ValueError('You must specify some parameters for search (e.g., ra, dec, radius)')
    checklegal(table,release)
    if format not in ('csv','votable','json'):
        raise ValueError('format must be csv, votable, or json')
    url = f"{BASEURL}/{release}/{table}.{format}"
    if columns:
        data['columns'] = '[{}]'.format(','.join(columns))
    r = requests.get(url, params=data, timeout=30)
    r.raise_for_status()
    if format == 'json':
        return r.json()
    else:
        return r.text

def ps1cone(ra, dec, radius_deg, table='mean', release='dr2', format='csv', columns=None, verbose=False, **kw):
    """Cone search of PS1 via MAST catalogs.
    ra, dec in degrees, radius_deg in degrees.
    Returns an astropy Table.
    """
    radius = float(radius_deg)
    params = kw.copy()
    params['ra'] = float(ra)
    params['dec'] = float(dec)
    params['radius'] = radius
    txt = ps1search(table=table, release=release, format=format, columns=columns, verbose=verbose, **params)
    # parse CSV into astropy Table
    tbl = ascii.read(txt)
    return tbl
