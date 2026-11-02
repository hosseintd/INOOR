# controllers/astrometry_workers.py
from PyQt5.QtCore import QThread, pyqtSignal
from core.astrometry.astrometry_solver import AstrometrySolver
from core.astrometry.catalog_query import ps1cone
from astropy.io import ascii
import os
import csv
import math
import time

import numpy as np
import requests


def _angular_sep_deg(ra1, dec1, ra2, dec2):
    """Angular separation in degrees (small-angle, cos(dec)-corrected).

    A plain planar sqrt(dRA^2 + dDec^2) overweights the RA difference by
    1/cos(dec) -- increasingly wrong at high declination.
    """
    dra = (float(ra1) - float(ra2) + 180.0) % 360.0 - 180.0  # wrap to [-180, 180]
    ddec = float(dec1) - float(dec2)
    cosd = math.cos(math.radians(0.5 * (float(dec1) + float(dec2))))
    return math.hypot(dra * cosd, ddec)

class PlateSolveWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    # Set by the controller's Cancel. Qt's interruption flag cannot carry this: it is
    # reset when the thread ends, before the queued result reaches the main thread.
    cancelled = False

    def __init__(self, fits_path, params):
        super().__init__()
        self.fits_path = fits_path
        self.params = params

    def run(self):
        try:
            solver = AstrometrySolver(
                api_key=self.params.get('api_key'),
                timeout=self.params.get('timeout', 900)
            )
            result = solver.solve_from_file(
                self.fits_path,
                scale_lower_arcmin=self.params.get('scale_low', 2.0),
                scale_upper_arcmin=self.params.get('scale_high', 10.0),
                downsample_factor=self.params.get('downsample', 4),
                ra=self.params.get('ra'),
                dec=self.params.get('dec'),
                search_radius=self.params.get('search_radius')
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class CatalogQueryWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, ra, dec, radius_arcsec=1.0):
        super().__init__()
        self.ra = ra
        self.dec = dec
        self.radius = radius_arcsec

    def run(self):
        try:
            radius_deg = self.radius / 3600.0
            # Query all available Pan-STARRS filters: u, g, r, i, z, y
            columns = ["objID","raMean","decMean","nDetections",
                      "uMeanPSFMag","gMeanPSFMag","rMeanPSFMag","iMeanPSFMag","zMeanPSFMag","yMeanPSFMag"]
            res = ps1cone(self.ra, self.dec, radius_deg, table='mean', release='dr2', format='csv', columns=columns)
            # res is astropy Table (returned by our migrated catalog_query.py).
            # Sort by true angular separation to the queried position so the
            # consumer's `table[0]` is the NEAREST source, not an arbitrary
            # catalog row (in crowded fields the first row could be a
            # different star than the one clicked).
            if len(res) > 1:
                seps = np.array([_angular_sep_deg(row['raMean'], row['decMean'], self.ra, self.dec)
                                 for row in res])
                res = res[np.argsort(seps)]
            self.finished.emit(res)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # The raw text ("HTTPSConnectionPool(...): Max retries exceeded ...") hid the cause.
            self.error.emit(f'Could not reach the PS1 catalog at MAST. Check the internet '
                            f'connection. ({type(e).__name__})')
        except Exception as e:
            self.error.emit(str(e))

class BatchExportWorker(QThread):
    progress = pyqtSignal(int, int) # Current, Total
    finished = pyqtSignal(str) # Path
    error = pyqtSignal(str)

    def __init__(self, stars, out_path, radius_arcsec=1.0):
        super().__init__()
        self.stars = stars
        self.out_path = out_path
        self.radius = radius_arcsec
        # One entry per star whose PS1 query raised. A bare except used to swallow these,
        # so with no network the file came out with every magnitude empty, as a success.
        self.failed_queries = []

    def run(self):
        try:
            rows_out = []
            total = len(self.stars)
            # Query all available Pan-STARRS filters: u, g, r, i, z, y
            columns = ["objID","raMean","decMean","nDetections",
                      "uMeanPSFMag","gMeanPSFMag","rMeanPSFMag","iMeanPSFMag","zMeanPSFMag","yMeanPSFMag"]
            
            for i, s in enumerate(self.stars):
                # Check if thread has been requested to stop
                if self.isInterruptionRequested():
                    # Emit finished even when cancelled: returning silently left
                    # the caller's progress dialog on screen forever, because
                    # on_export_finished is what closes it.
                    self.finished.emit("")
                    return
                
                ra = s.get('ra_deg')
                dec = s.get('dec_deg')
                
                ps1_row = None
                if ra is not None and dec is not None:
                    try:
                        res = ps1cone(ra, dec, self.radius / 3600.0, columns=columns)
                        if len(res) > 0:
                            # Closest match by true angular separation
                            best_row = None; best_sep = 1e9
                            for row in res:
                                sep = _angular_sep_deg(row['raMean'], row['decMean'], ra, dec)
                                if sep < best_sep:
                                    best_sep = sep; best_row = row
                            ps1_row = best_row
                    except Exception as e:
                        self.failed_queries.append(f'{type(e).__name__}: {e}')
                
                
                # The band columns carry a _ps1 suffix: a plain 'y' key for the PS1 y band
                # overwrote the pixel y, so every export had the y magnitude under 'y'.
                # PS1 has no u band (that column was always empty).
                row_dict = {'x': s['x'], 'y': s['y'], 'ra_deg': ra, 'dec_deg': dec}
                for band in 'grizy':
                    col = f'{band}MeanPSFMag'
                    row_dict[f'{band}_ps1'] = ps1_row[col] if ps1_row and col in ps1_row.colnames else ''
                rows_out.append(row_dict)
                self.progress.emit(i+1, total)

            # Check again before writing file
            if self.isInterruptionRequested():
                self.finished.emit("")
                return

            with open(self.out_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_dict.keys())
                writer.writeheader()
                writer.writerows(rows_out)
            
            self.finished.emit(self.out_path)
        except Exception as e:
            self.error.emit(str(e))
