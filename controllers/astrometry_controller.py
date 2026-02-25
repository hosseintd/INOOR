# controllers/astrometry_controller.py
import os
import time
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from .astrometry_workers import PlateSolveWorker, CatalogQueryWorker, BatchExportWorker
from utils.astrometry_config import SCALE_PRESETS, get_export_dir

class AstrometryController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view
        self._setup_connections()

    def _setup_connections(self):
        self.view.open_fits_clicked.connect(self.on_open_fits)
        self.view.control_panel.solve_clicked.connect(self.on_solve_plate)
        self.view.control_panel.preset_changed.connect(self.on_preset_changed)
        self.view.control_panel.view_action.connect(self.on_view_action)
        self.view.control_panel.cmap_changed.connect(self.view.viewer.set_cmap)
        self.view.control_panel.stretch_changed.connect(self.on_stretch_changed)
        
        self.view.btn_query_ps1.clicked.connect(self.on_query_ps1)
        self.view.btn_export_csv.clicked.connect(self.on_export_csv)

    @pyqtSlot()
    def on_open_fits(self):
        path, _ = QFileDialog.getOpenFileName(None, "Open FITS for Astrometry", 
                                               filter="FITS files (*.fits *.fit *.fits.gz)")
        if path:
            self.model.set_current_file(path)
            self.view.viewer.load_fits(path)

    @pyqtSlot(str)
    def on_preset_changed(self, preset):
        if preset in SCALE_PRESETS:
            low, high = SCALE_PRESETS[preset]
            self.view.control_panel.set_scales(low, high)

    @pyqtSlot(dict)
    def on_solve_plate(self, params):
        if not self.model.current_file:
            QMessageBox.warning(None, "No Image", "Please load a FITS image first.")
            return
        
        self.progress = QProgressDialog("Solving plate via Astrometry.net...", "Cancel", 0, 0, self.view)
        self.progress.setWindowTitle("Astrometry Solving")
        self.progress.show()

        self.worker = PlateSolveWorker(self.model.current_file, params)
        self.worker.finished.connect(self.on_solve_finished)
        self.worker.error.connect(self.on_solve_error)
        self.worker.start()

    def on_solve_finished(self, result):
        self.progress.close()
        self.model.update_solve_result(result)
        
        solved_path = result.get('solved_fits')
        if solved_path and os.path.exists(solved_path):
            # Safety check: if file is 0 bytes, it's a failed download/solve
            if os.path.getsize(solved_path) == 0:
                QMessageBox.warning(None, "Solve Result Empty", "Astrometry.net returned an empty solved file. This usually happens when the solve fails on the server side.")
                return
            
            try:
                self.view.viewer.load_fits(solved_path)
                self.view.viewer.last_downsample_factor = int(result.get('downsample_factor', 1))
            except Exception as e:
                QMessageBox.critical(None, "Load Error", f"Failed to load the solved FITS file: {e}\nThe file may be corrupted.")
                return
            
            # Load annotations if available
            rdls = result.get('rdls_path')
            axy = result.get('axy_path')
            if (rdls and os.path.exists(rdls)) or (axy and os.path.exists(axy)):
                self.view.viewer.load_astrometry_files(rdls, axy)
            
            QMessageBox.information(None, "Success", f"Plate solved successfully.\nWCS written to: {os.path.basename(solved_path)}")
        else:
            err = result.get('error')
            msg = f"Astrometry.net could not solve the image or return results.\nError: {err}" if err else "Astrometry.net could not solve the image or return results."
            if not err or err.strip() == '':
                # If error is empty, check if there's a WCS application issue
                msg = "Astrometry.net could not solve the image. Check the console for detailed error information."
            QMessageBox.warning(None, "Solve Failed", msg)

    def on_solve_error(self, err_msg):
        self.progress.close()
        QMessageBox.critical(None, "Astrometry Error", f"Failed to solve: {err_msg}")

    @pyqtSlot(str)
    def on_view_action(self, action):
        if action == 'rotate_cw': self.view.viewer.rotate90(True)
        elif action == 'rotate_ccw': self.view.viewer.rotate90(False)
        elif action == 'flip_h': self.view.viewer.flip_horizontal()
        elif action == 'flip_v': self.view.viewer.flip_vertical()
        elif action == 'toggle_det': 
            self.view.viewer.toggle_detected()

    @pyqtSlot(str, float, float)
    def on_stretch_changed(self, mode, vmin, vmax):
        if mode == 'zscale':
            self.view.viewer.set_stretch_zscale()
        else:
            self.view.viewer.set_stretch_minmax(vmin, vmax)

    @pyqtSlot()
    def on_query_ps1(self):
        sky = getattr(self.view.viewer, 'last_picked_world', None)
        if not sky:
            QMessageBox.warning(None, "Selection", "Click a star marker first.")
            return
        
        self.view.info_box.append("\nQuerying PS1 catalog...")
        self.cat_worker = CatalogQueryWorker(sky.ra.deg, sky.dec.deg)
        self.cat_worker.finished.connect(self.on_query_finished)
        self.cat_worker.error.connect(lambda e: self.view.info_box.append(f"Query error: {e}"))
        self.cat_worker.start()

    def on_query_finished(self, table):
        if len(table) == 0:
            self.view.info_box.append("No PS1 match found.")
            return
        
        # Closest match
        row = table[0] # Simplification, workers does more if we wanted
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        sc = SkyCoord(row['raMean'], row['decMean'], unit=u.deg)
        
        res = [
            f"--- PS1 Match ---",
            f"RA: {sc.ra.deg:.8f}, Dec: {sc.dec.deg:.8f}",
            f"u: {row['uMeanPSFMag']:.3f}" if 'uMeanPSFMag' in row.colnames else "",
            f"g: {row['gMeanPSFMag']:.3f}" if 'gMeanPSFMag' in row.colnames else "",
            f"r: {row['rMeanPSFMag']:.3f}" if 'rMeanPSFMag' in row.colnames else "",
            f"i: {row['iMeanPSFMag']:.3f}" if 'iMeanPSFMag' in row.colnames else "",
            f"z: {row['zMeanPSFMag']:.3f}" if 'zMeanPSFMag' in row.colnames else "",
            f"y: {row['yMeanPSFMag']:.3f}" if 'yMeanPSFMag' in row.colnames else "",
        ]
        self.view.info_box.setText(self.view.info_box.toPlainText() + "\n" + "\n".join(filter(None, res)))

    @pyqtSlot()
    def on_export_csv(self):
        all_stars = self.view.viewer.export_annotated_list() # list of dicts with ra_deg, dec_deg, kind...
        if not all_stars:
            QMessageBox.information(None, "Export", "No stars to export.")
            return
        
        # Filter to only include detected sources (green markers, kind='det')
        green_stars = [s for s in all_stars if s.get('kind') == 'det']
        if not green_stars:
            QMessageBox.information(None, "Export", "No detected (green) stars to export.")
            return
            
        out_dir = get_export_dir()
        fname = f"astrometry_export_{int(time.time())}.csv"
        path = os.path.join(out_dir, fname)
        
        self.exp_prog = QProgressDialog("Batch querying PS1 for export (detected sources only)...", "Cancel", 0, len(green_stars), self.view)
        self.exp_prog.show()
        
        self.exp_worker = BatchExportWorker(green_stars, path)
        self.exp_worker.progress.connect(lambda c, t: self.exp_prog.setValue(c))
        self.exp_worker.finished.connect(self.on_export_finished)
        self.exp_worker.error.connect(lambda e: QMessageBox.critical(None, "Export Error", e))
        # Connect cancel button to stop the worker
        self.exp_prog.canceled.connect(self.exp_worker.requestInterruption)
        self.exp_worker.start()

    def on_export_finished(self, path):
        self.exp_prog.close()
        QMessageBox.information(None, "Export Success", f"Exported to: {path}")
