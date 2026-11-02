# controllers/astrometry_controller.py
import os
import time
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from .astrometry_workers import PlateSolveWorker, CatalogQueryWorker, BatchExportWorker
from utils.astrometry_config import SCALE_PRESETS, get_export_dir
from utils.thread_guard import is_worker_busy

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
            # The box still described a star of the previous frame.
            self.view.info_box.clear()

    @pyqtSlot(str)
    def on_preset_changed(self, preset):
        if preset in SCALE_PRESETS:
            low, high = SCALE_PRESETS[preset]
            self.view.control_panel.set_scales(low, high)

    @pyqtSlot(dict)
    def on_solve_plate(self, params):
        if is_worker_busy(self, 'worker', view=self.view,
                          busy_msg="A plate-solve is already in progress."):
            return
        if not self.model.current_file:
            QMessageBox.warning(None, "No Image", "Please load a FITS image first.")
            return

        self.progress = QProgressDialog("Solving plate via Astrometry.net...", "Cancel", 0, 0, self.view)
        self.progress.setWindowTitle("Astrometry Solving")
        self.progress.show()

        self.worker = PlateSolveWorker(self.model.current_file, params)
        # The handlers get the worker that finished, so a cancelled one can be recognised.
        worker = self.worker
        worker.finished.connect(lambda result: self.on_solve_finished(result, worker))
        worker.error.connect(lambda err_msg: self.on_solve_error(err_msg, worker))
        # The Cancel button was never connected to anything, so it looked
        # live and did nothing on a job that can run for 900 s.
        self.progress.canceled.connect(lambda: self._cancel_solve(worker))
        self.worker.start()

    def _cancel_solve(self, worker):
        worker.cancelled = True
        worker.requestInterruption()
        self.view.info_box.append("Plate solve cancelled. Astrometry.net may still finish "
                                  "the job; its result will be ignored.")

    def _close_progress(self):
        # QProgressDialog.close() emits canceled(), which reported every finished solve
        # as cancelled.
        self.progress.canceled.disconnect()
        self.progress.close()

    def on_solve_finished(self, result, worker=None):
        # Cancel cannot stop astrometry.net, so the worker still returns a result.
        # Applying it put the cancelled frame's solution on screen (with "Success")
        # even after the operator had opened another frame.
        if worker is not None and worker.cancelled:
            return
        self._close_progress()
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
            
            QMessageBox.information(None, "Success", f"Plate solved successfully.\nWCS written to:\n{os.path.abspath(solved_path)}")
        else:
            err = result.get('error')
            msg = f"Astrometry.net could not solve the image or return results.\nError: {err}" if err else "Astrometry.net could not solve the image or return results."
            if not err or err.strip() == '':
                # If error is empty, check if there's a WCS application issue
                msg = "Astrometry.net could not solve the image. Check the console for detailed error information."
            QMessageBox.warning(None, "Solve Failed", msg)

    def on_solve_error(self, err_msg, worker=None):
        if worker is not None and worker.cancelled:
            return
        self._close_progress()
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
        
        if is_worker_busy(self, 'cat_worker'):
            self.view.info_box.append("\nPrevious PS1 query still running...")
            return

        self.view.info_box.append("\nQuerying PS1 catalog...")
        self.cat_worker = CatalogQueryWorker(sky.ra.deg, sky.dec.deg)
        # The query takes seconds. A frame opened meanwhile must not get this star's magnitudes.
        frame = self.model.current_file
        self.cat_worker.finished.connect(lambda table: self.on_query_finished(table, frame))
        self.cat_worker.error.connect(lambda e: self.on_query_error(e, frame))
        self.cat_worker.start()

    def on_query_error(self, err, frame=None):
        if frame is not None and frame != self.model.current_file:
            return
        self.view.info_box.append(f"Query error: {err}")

    def on_query_finished(self, table, frame=None):
        if frame is not None and frame != self.model.current_file:
            return
        if len(table) == 0:
            self.view.info_box.append("No PS1 match found.")
            return
        
        # CatalogQueryWorker sorts the table by angular separation to the
        # clicked position, so row 0 is the nearest catalog source.
        row = table[0]
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
        if is_worker_busy(self, 'exp_worker', view=self.view,
                          busy_msg="A CSV export is already in progress."):
            return
        all_stars = self.view.viewer.export_annotated_list() # list of dicts with ra_deg, dec_deg, kind...
        if not all_stars:
            QMessageBox.information(None, "Export", "No stars to export.")
            return
        
        # Filter to only include detected sources (red markers, kind='det')
        detected = [s for s in all_stars if s.get('kind') == 'det']
        if not detected:
            QMessageBox.information(None, "Export", "No detected stars to export.")
            return
            
        out_dir = get_export_dir()
        fname = f"astrometry_export_{int(time.time())}.csv"
        path = os.path.join(out_dir, fname)
        
        self.exp_prog = QProgressDialog("Batch querying PS1 for export (detected sources only)...", "Cancel", 0, len(detected), self.view)
        self.exp_prog.show()

        self.exp_worker = BatchExportWorker(detected, path)
        self.exp_worker.progress.connect(lambda c, t: self.exp_prog.setValue(c))
        self.exp_worker.finished.connect(self.on_export_finished)
        self.exp_worker.error.connect(self.on_export_error)
        # Connect cancel button to stop the worker
        self.exp_prog.canceled.connect(self.exp_worker.requestInterruption)
        self.exp_worker.start()

    def on_export_error(self, err):
        self.exp_prog.close()
        QMessageBox.critical(None, "Export Error", err)

    def on_export_finished(self, path):
        self.exp_prog.close()
        if not path:
            # Cancelled: the worker emits an empty path so this handler still
            # runs and closes the dialog.
            QMessageBox.information(None, "Export Cancelled",
                                    "The catalogue export was cancelled; no file "
                                    "was written.")
            return
        failed = self.exp_worker.failed_queries
        if failed:
            QMessageBox.warning(
                None, "Export Incomplete",
                f"Exported to: {path}\n\nThe PS1 query failed for {len(failed)} of "
                f"{len(self.exp_worker.stars)} stars (no network, or MAST unavailable), "
                f"so their magnitude columns are empty.\n\nFirst error: {failed[0][:300]}")
            return
        QMessageBox.information(None, "Export Success", f"Exported to: {path}")
