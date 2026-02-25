import os
import sys
import numpy as np
from astropy.io import fits
from PyQt5.QtCore import QObject, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox

# Adjust path to find root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models.file_set import FileSet
from views.create_master_view import CreateMasterView
from views.components.preview_dialog import PreviewDialog
from core import masterFrame_creator as mfc
from utils.exptime_utils import format_exptime
from controllers.workers import CreateMasterWorker, HistogramWorker, FitWorker

class CreateMasterController(QObject):
    def __init__(self, model, view: CreateMasterView):
        super().__init__()
        self.model = model
        self.view = view
        
        # --- Connect View signals ---
        self.view.add_set_clicked.connect(self.add_to_main_set) 
        self.view.remove_files_clicked.connect(self.remove_selected_files)
        self.view.remove_all_files_clicked.connect(self.remove_all_files)
        self.view.file_selected.connect(self.select_file)
        
        self.view.param_changed.connect(self.update_params_from_view)
        self.view.toggle_bad_frame.connect(self.toggle_bad_frame)
        self.view.create_master_clicked.connect(self.create_master)
        
        self.view.estimate_sigma_clicked.connect(self.estimate_sigma)
        self.view.show_full_header_clicked.connect(self.show_full_header)
        self.view.hist_update_range_clicked.connect(self.update_hist_range)
        self.view.hist_reset_clicked.connect(self.reset_hist_range)
        self.view.hist_bitdepth_changed.connect(self.on_bitdepth_changed)
        
        # --- Connect Model signals ---
        self.model.sets_changed.connect(self.on_sets_changed)
        self.model.data_changed.connect(self.on_data_changed)
        
        # Initial State
        self.current_preview_file = None
        self.current_img_data = None
        
        # If no sets exist, create a default one to mimic legacy behavior
        if not self.model.get_sets():
            self.model.add_set(FileSet(name="Main Set", set_type="Light"))

        self.refresh_all()

    def refresh_all(self):
        self.on_sets_changed()
        self.on_data_changed()

    @pyqtSlot()
    def on_sets_changed(self):
        # In this legacy-look version, we only show files from the current active set
        fs = self.model.get_current_set()
        self.view.update_file_list(fs)

    @pyqtSlot()
    def on_data_changed(self):
        fs = self.model.get_current_set()
        self.view.update_params(fs)
        self.view.update_file_list(fs)

    @pyqtSlot(str, list)
    def add_to_main_set(self, set_type, paths):
        fs = self.model.get_current_set()
        if not fs:
            fs = FileSet(set_type=set_type, files=paths)
            self.model.add_set(fs)
        else:
            # Prevent duplicates
            new_files = [p for p in paths if p not in fs.files]
            fs.files.extend(new_files)
        
        self.model.trigger_update()

    @pyqtSlot()
    def remove_selected_files(self):
        fs = self.model.get_current_set()
        if not fs: return
        indices = self.view.get_selected_file_indices()
        if not indices: return
        
        for i in sorted(indices, reverse=True):
            if i < len(fs.files):
                fs.files.pop(i)
                if i in fs.bad_indices:
                    fs.bad_indices.remove(i)
                # Re-shift bad indices
                new_bad = set()
                for bi in fs.bad_indices:
                    if bi > i: new_bad.add(bi - 1)
                    else: new_bad.add(bi)
                fs.bad_indices = new_bad
        
        self.model.trigger_update()
        if not fs.files:
            self.view.canvas_image.show_image(None)
            self.view.canvas_hist.show_hist(None, None)
            self.view.clear_stats()

    @pyqtSlot()
    def remove_all_files(self):
        fs = self.model.get_current_set()
        if not fs: return
        
        reply = QMessageBox.question(self.view, "Confirm", "Remove all files from this set?", 
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No: return
        
        fs.files.clear()
        fs.bad_indices.clear()
        self.model.trigger_update()
        self.view.canvas_image.show_image(None)
        self.view.canvas_hist.show_hist(None, None)
        self.view.clear_stats()

    @pyqtSlot(int)
    def select_file(self, row):
        fs = self.model.get_current_set()
        if not fs or row < 0 or row >= len(fs.files):
            return
            
        path = fs.files[row]
        self.current_preview_file = path
        
        try:
            img = mfc.load_fits(path)
            self.current_img_data = img
            
            # Update View
            self.view.canvas_image.show_image(img)
            self.view.lbl_filename.setText(os.path.basename(path))
            
            # ZScale initialization
            vmin, vmax = mfc.zscale_limits(img)
            self.view.update_hist_spinboxes(vmin, vmax)
            
            # Statistics
            arr = img.ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                stats = {
                    'mean': np.mean(arr),
                    'median': np.median(arr),
                    'std': np.std(arr),
                    'min': np.min(arr),
                    'max': np.max(arr),
                    'rms': np.sqrt(np.mean(arr**2))
                }
                self.view.set_stats(stats)
            
            # Histogram
            sample_size = 200000
            if arr.size > sample_size:
                step = arr.size // sample_size
                sample = arr[::step]
            else:
                sample = arr
            counts, bins = np.histogram(sample, bins=2000)
            self.view.canvas_hist.show_hist(counts, bins)
            
            # Header Info
            with fits.open(path) as hdul:
                hdr = hdul[0].header
                # Use format_exptime to properly format exposure time with appropriate units
                exptime_str = format_exptime(hdr)
                self.view.set_header_info({
                    'EXPTIME': exptime_str,
                    'CCD-TEMP': hdr.get('CCD-TEMP', 'N/A'),
                    'YBINNING': hdr.get('YBINNING', 'N/A'),
                    'XBINNING': hdr.get('XBINNING', 'N/A'),
                    'GAIN': hdr.get('GAIN', 'N/A'),
                    'DATE-OBS': hdr.get('DATE-OBS', 'N/A')
                })
            
            # Bad frame status
            self.view.set_bad_checkbox(row in fs.bad_indices)
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            self.view.lbl_filename.setText("Error loading file")

    @pyqtSlot(int, bool)
    def toggle_bad_frame(self, index, is_bad):
        fs = self.model.get_current_set()
        if not fs: return
        if is_bad:
            fs.bad_indices.add(index)
        else:
            fs.bad_indices.discard(index)
        self.model.trigger_update()

    @pyqtSlot()
    def update_params_from_view(self):
        fs = self.model.get_current_set()
        if not fs: return
        
        fs.method = 'mean' if self.view.radio_mean.isChecked() else 'median'
        fs.do_sigma_clip = self.view.chk_sigma.isChecked()
        fs.auto_sigma = self.view.radio_sig_auto.isChecked()
        fs.sigma_lower = self.view.spin_sigma_lo.value()
        fs.sigma_upper = self.view.spin_sigma_hi.value()
        # Dynamically add kernel size to the set model (or handle it in worker)
        fs.sigma_kernel = self.view.spin_sigma_kernel.value() 
        fs.create_gain_table = self.view.chk_gain.isChecked()
        fs.gain_poly_degree = self.view.spin_degree.value()

    @pyqtSlot()
    def create_master(self):
        fs = self.model.get_current_set()
        if not fs or not fs.files:
            QMessageBox.warning(self.view, "No files", "Please add frames first.")
            return
        
        # Sync latest params from view just in case
        self.update_params_from_view()
        
        self.view.lbl_status.setText("Stacking Master... Please wait.")
        self.view.btn_create_master.setEnabled(False)
        
        # Sigma Clipping parameters
        lower = fs.sigma_lower
        upper = fs.sigma_upper
        kernel = getattr(fs, 'sigma_kernel', 5)
        auto_sigma = getattr(fs, 'auto_sigma', False)
        
        # Pass out_file=None to triggering the Review mode in the worker
        self.worker = CreateMasterWorker(
            out_file=None, 
            file_list=fs.files,
            method=fs.method,
            do_clip=fs.do_sigma_clip,
            lower=lower,
            upper=upper,
            excludes=list(fs.bad_indices),
            kernel=kernel,
            auto_sigma=auto_sigma
        )
        self.worker.progress.connect(lambda f, m: self.view.lbl_status.setText(f"{m} ({f*100:.0f}%)"))
        self.worker.finished.connect(self.on_master_ready_for_review)
        self.worker.start()

    @pyqtSlot(bool, object, object, str)
    def on_master_ready_for_review(self, success, data, header, msg):
        self.view.btn_create_master.setEnabled(True)
        self.view.lbl_status.setText("Ready")
        
        if not success:
            QMessageBox.critical(self.view, "Error", f"Failed to create master:\n{msg}")
            return

        # Show Review Window
        dlg = PreviewDialog(data, title="Review Created Master (ZScale)", parent=self.view, zscale=True)
        if dlg.exec_() == QMessageBox.Accepted:
            # User confirmed, now ask where to save
            default_name = self._generate_default_master_name(header)
            out_path, _ = QFileDialog.getSaveFileName(self.view, "Save Master", default_name, "FITS (*.fits)")
            
            if out_path:
                try:
                    # Save using the helper from mfc, which handles N-bit conversion and header
                    # PASS HEADER OBJECT DIRECTLY (to avoid \n issues with dict conversion)
                    mfc.write_nbit_fits(out_path, data, nbits=12, hdr_extra=header, overwrite=True)
                    QMessageBox.information(self.view, "Success", f"Master saved:\n{out_path}")
                    
                    # Proceed to Gain Table if requested
                    fs = self.model.get_current_set()
                    if fs and fs.create_gain_table:
                        self.start_gain_table_workflow(out_path, fs.gain_poly_degree)
                except Exception as e:
                    QMessageBox.critical(self.view, "Save Error", str(e))
        else:
            self.view.lbl_status.setText("Master creation rejected.")

    def _generate_default_master_name(self, header):
        import re
        fs = self.model.get_current_set()
        if not fs or not fs.files:
            return "Master_Frame.fits"
            
        first_file = os.path.basename(fs.files[0])
        
        # 1. Detect Frame Type
        frame_type = "Frame"
        if fs.create_gain_table:
            frame_type = "Gain_Table"
        else:
            # Heuristic detection from filename
            fname_lower = first_file.lower()
            if "dark" in fname_lower: frame_type = "Dark"
            elif "bias" in fname_lower: frame_type = "Bias"
            elif "flat" in fname_lower: frame_type = "Flat"
            elif "light" in fname_lower: frame_type = "Light"
        
        # 2. Extract Exposure Time
        exp_suffix = ""
        # Try filename regex first e.g. exp00.00.30.000
        match = re.search(r"exp(\d+)\.(\d+)\.(\d+)\.(\d+)", first_file)
        if match:
            h, m, s, ms = map(int, match.groups())
            total_sec = h * 3600 + m * 60 + s
            if total_sec > 0:
                exp_suffix = f"_{total_sec}s"
        else:
            # Try FITS header EXPTIME
            et = header.get('EXPTIME')
            if et is not None:
                try:
                    val = float(et)
                    if val > 0:
                        exp_suffix = f"_{int(val)}s"
                except: pass
                
        # 3. Sigma Clipping Suffix
        sig_suffix = ""
        if fs.do_sigma_clip:
            mode = "auto" if getattr(fs, 'auto_sigma', False) else "manual"
            kernel = getattr(fs, 'sigma_kernel', 5)
            sig_suffix = f"_{mode}_sigclipped_kernelsize-{kernel}"
            
        return f"Master_{frame_type}{exp_suffix}{sig_suffix}.fits"

    def start_gain_table_workflow(self, master_path, degree):
        try:
            arr = mfc.load_fits(master_path)
            self.fit_worker = FitWorker(arr, degree=degree)
            self.fit_worker.done.connect(self.on_fit_done)
            self.fit_worker.failed.connect(lambda m: QMessageBox.critical(self.view, "Fit error", m))
            self.view.lbl_status.setText("Fitting polynomial...")
            self.fit_worker.start()
        except Exception as e:
            QMessageBox.critical(self.view, "Error", str(e))

    def on_fit_done(self, z_fitted, gain_table, coeffs):
        self.view.lbl_status.setText("Ready")
        # Preview dialogs like in legacy
        dlg = PreviewDialog(z_fitted, title="Fitted Surface Preview", parent=self.view)
        if dlg.exec_() == QMessageBox.Accepted:
            dlg2 = PreviewDialog(gain_table, title="Gain Table Preview", parent=self.view)
            if dlg2.exec_() == QMessageBox.Accepted:
                save_path, _ = QFileDialog.getSaveFileName(self.view, "Save Gain Table", "gain_table.fits", "FITS (*.fits)")
                if save_path:
                    hdu = fits.PrimaryHDU(gain_table.astype('float32'))
                    hdu.writeto(save_path, overwrite=True)
                    QMessageBox.information(self.view, "Saved", f"Gain table saved to {save_path}")

    @pyqtSlot()
    def estimate_sigma(self):
        if self.current_img_data is None: return
        l, u = mfc.analyze_sigma_bounds(self.current_img_data)
        self.view.spin_sigma_lo.setValue(l)
        self.view.spin_sigma_hi.setValue(u)

    @pyqtSlot()
    def show_full_header(self):
        if not self.current_preview_file: return
        try:
            from astropy.io import fits
            hdr = fits.getheader(self.current_preview_file)
            from PyQt5 import QtWidgets, QtGui
            
            dlg = QtWidgets.QDialog(self.view)
            dlg.setWindowTitle(f"Header: {os.path.basename(self.current_preview_file)}")
            dlg.resize(700, 800)
            layout = QtWidgets.QVBoxLayout(dlg)
            
            # Search Bar
            search_box = QtWidgets.QHBoxLayout()
            search_label = QtWidgets.QLabel("Search Keywords:")
            search_input = QtWidgets.QLineEdit()
            search_box.addWidget(search_label)
            search_box.addWidget(search_input)
            layout.addLayout(search_box)
            
            # Text Area
            text_edit = QtWidgets.QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setAcceptRichText(True)
            text_edit.setFont(QtGui.QFont("Courier New", 10))
            layout.addWidget(text_edit)
            
            # Format header string
            cards = []
            for card in hdr.cards:
                cards.append(str(card))
            full_text = "\n".join(cards)
            text_edit.setPlainText(full_text)
            
            def perform_search(text):
                text_edit.setPlainText(full_text)
                if not text: return
                
                cursor = text_edit.textCursor()
                format = QtGui.QTextCharFormat()
                format.setBackground(QtGui.QBrush(QtGui.QColor("yellow")))
                
                doc = text_edit.document()
                cursor = QtGui.QTextCursor(doc)
                
                while not cursor.isNull() and not cursor.atEnd():
                    cursor = doc.find(text, cursor)
                    if not cursor.isNull():
                        cursor.mergeCharFormat(format)
            
            search_input.textChanged.connect(perform_search)
            
            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(dlg.accept)
            layout.addWidget(close_btn)
            
            dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self.view, "Error", f"Could not show header: {e}")

    @pyqtSlot(float, float)
    def update_hist_range(self, mn, mx):
        self.view.canvas_image.set_clim(mn, mx)

    @pyqtSlot()
    def reset_hist_range(self):
        if self.current_img_data is None: return
        vmin, vmax = mfc.zscale_limits(self.current_img_data)
        self.update_hist_range(vmin, vmax)
        self.view.update_hist_spinboxes(vmin, vmax)

    @pyqtSlot(int)
    def on_bitdepth_changed(self, bits):
        # View already handles range update signals if needed via spinboxes
        # But we don't have a slider anymore to update.
        pass

