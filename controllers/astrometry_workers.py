# controllers/astrometry_workers.py
from PyQt5.QtCore import QThread, pyqtSignal
from core.astrometry.astrometry_solver import AstrometrySolver
from core.astrometry.catalog_query import ps1cone
from astropy.io import ascii
import os
import csv
import time

class PlateSolveWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
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
            # res is astropy Table (returned by our migrated catalog_query.py)
            self.finished.emit(res)
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
                    print("CSV export cancelled by user")
                    return
                
                ra = s.get('ra_deg')
                dec = s.get('dec_deg')
                
                ps1_row = None
                if ra is not None and dec is not None:
                    try:
                        res = ps1cone(ra, dec, self.radius / 3600.0, columns=columns)
                        if len(res) > 0:
                            # Simple closest match
                            best_row = None; best_sep = 1e9
                            for row in res:
                                sep = ((float(row['raMean'])-ra)**2 + (float(row['decMean'])-dec)**2)**0.5
                                if sep < best_sep:
                                    best_sep = sep; best_row = row
                            ps1_row = best_row
                    except: pass
                
                
                row_dict = {
                    'x': s['x'], 'y': s['y'], 'ra_deg': ra, 'dec_deg': dec,
                    'u': ps1_row['uMeanPSFMag'] if ps1_row and 'uMeanPSFMag' in ps1_row.colnames else '',
                    'g': ps1_row['gMeanPSFMag'] if ps1_row and 'gMeanPSFMag' in ps1_row.colnames else '',
                    'r': ps1_row['rMeanPSFMag'] if ps1_row and 'rMeanPSFMag' in ps1_row.colnames else '',
                    'i': ps1_row['iMeanPSFMag'] if ps1_row and 'iMeanPSFMag' in ps1_row.colnames else '',
                    'z': ps1_row['zMeanPSFMag'] if ps1_row and 'zMeanPSFMag' in ps1_row.colnames else '',
                    'y': ps1_row['yMeanPSFMag'] if ps1_row and 'yMeanPSFMag' in ps1_row.colnames else '',
                }
                rows_out.append(row_dict)
                self.progress.emit(i+1, total)

            # Check again before writing file
            if self.isInterruptionRequested():
                print("CSV export cancelled by user")
                return

            with open(self.out_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_dict.keys())
                writer.writeheader()
                writer.writerows(rows_out)
            
            self.finished.emit(self.out_path)
        except Exception as e:
            self.error.emit(str(e))
