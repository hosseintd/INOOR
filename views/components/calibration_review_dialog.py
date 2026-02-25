from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import os
from views.components.zoom_image_canvas import ZoomImageCanvas
from core import photometry_core as pc

class CalibrationReviewDialog(QtWidgets.QDialog):
    """
    Dialog to review detected reference stars.
    Features:
    - Next/Prev frame navigation
    - Visual inspection of detected star
    - Manual re-selection by clicking on canvas
    - Parameter override for difficult stars
    - Updates the parent's calibration history object directly
    """
    def __init__(self, parent, file_list, calibration_results_list, active_col_idx=0):
        super().__init__(parent)
        self.setWindowTitle("Review Calibration Stars")
        self.resize(1000, 750)
        
        self.files = file_list
        self.history_item = calibration_results_list[active_col_idx]
        self.results_map = self.history_item.get('results', {})
        self.col_name = self.history_item.get('name', f"Ref Star {active_col_idx+1}")
        
        self._current_idx = 0
        self.current_data = None
        
        self._setup_ui()
        self.load_frame(0)

    def _setup_ui(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        
        # Header Info
        top_layout = QtWidgets.QHBoxLayout()
        info = QtWidgets.QLabel(f"Reviewing: {self.col_name}")
        info.setStyleSheet("font-weight: bold; color: yellow; font-size: 16px;")
        top_layout.addWidget(info)
        top_layout.addStretch()
        self.layout.addLayout(top_layout)
        
        # Main Content: Canvas + Sidebar
        content_layout = QtWidgets.QHBoxLayout()
        
        # Canvas
        self.canvas = ZoomImageCanvas(self)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Connect click event for manual re-selection
        self.canvas.click_coords.connect(self.on_canvas_click)
        content_layout.addWidget(self.canvas, stretch=3)
        
        # Sidebar for Parms & Actions
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("background: #0f1a1a; border-left: 1px solid #2e6f6f;")
        sb_layout = QtWidgets.QVBoxLayout(sidebar)
        
        sb_layout.addWidget(QtWidgets.QLabel("Manual Re-Measure"))
        
        # Params Group
        grp_params = QtWidgets.QGroupBox("Photometry Params")
        form = QtWidgets.QFormLayout()
        
        self.spin_fwhm = QtWidgets.QDoubleSpinBox(); self.spin_fwhm.setValue(12.0)
        self.spin_thresh = QtWidgets.QDoubleSpinBox(); self.spin_thresh.setValue(3.0)
        self.spin_aperture = QtWidgets.QSpinBox(); self.spin_aperture.setRange(1, 100); self.spin_aperture.setValue(10)
        self.spin_inner = QtWidgets.QDoubleSpinBox(); self.spin_inner.setRange(1.1, 10); self.spin_inner.setValue(2.0)
        self.spin_outer = QtWidgets.QDoubleSpinBox(); self.spin_outer.setRange(1.2, 20); self.spin_outer.setValue(3.0)
        self.chk_fixed_ap = QtWidgets.QCheckBox("Fixed Aperture")
        
        form.addRow("FWHM:", self.spin_fwhm)
        form.addRow("Threshold:", self.spin_thresh)
        form.addRow("Aperture:", self.spin_aperture)
        form.addRow("Inner Coeff:", self.spin_inner)
        form.addRow("Outer Coeff:", self.spin_outer)
        form.addRow("", self.chk_fixed_ap)
        grp_params.setLayout(form)
        sb_layout.addWidget(grp_params)
        
        self.lbl_status = QtWidgets.QLabel("Click on star to re-select")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #d9f0ec; font-style: italic; margin-top: 10px;")
        sb_layout.addWidget(self.lbl_status)
        
        sb_layout.addStretch()
        content_layout.addWidget(sidebar)
        
        self.layout.addLayout(content_layout)
        
        # Navigation Bar
        nav = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("<< Previous")
        self.lbl_frame = QtWidgets.QLabel("Frame 1/N")
        self.lbl_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_frame.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_next = QtWidgets.QPushButton("Next >>")
        
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.lbl_frame)
        nav.addWidget(self.btn_next)
        self.layout.addLayout(nav)
        
        # Footer Actions
        footer = QtWidgets.QHBoxLayout()
        self.btn_outliers = QtWidgets.QPushButton("Identify Outliers")
        self.btn_close = QtWidgets.QPushButton("Done / Close")
        footer.addWidget(self.btn_outliers)
        footer.addStretch()
        footer.addWidget(self.btn_close)
        self.layout.addLayout(footer)
        
        # Connections
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_outliers.clicked.connect(self.identify_outliers)
        self.btn_close.clicked.connect(self.accept)
        
        self.setStyleSheet("""
            QWidget { background: #162a2a; color: #d9f0ec; }
            QPushButton { background: #0f7a73; color: white; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background: #148f87; }
            QGroupBox { border: 1px solid #2e6f6f; font-weight: bold; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QDoubleSpinBox, QSpinBox { background: #0e1a1a; border: 1px solid #2e6f6f; padding: 2px; }
        """)

    def identify_outliers(self):
        # Scan results for potential outliers (e.g. mag deviates > 2 sigma from median)
        mags = []
        indices = []
        for idx in range(len(self.files)):
            res = self.results_map.get(idx, {})
            if res and res.get('success'):
                 m = res.get('mag')
                 if m is not None:
                     mags.append(m)
                     indices.append(idx)
        
        if not mags:
             QtWidgets.QMessageBox.information(self, "No Data", "No valid magnitudes to check.")
             return
             
        med = np.median(mags)
        std = np.std(mags)
        
        # Simple Z-score check
        outliers = []
        for i, m, idx in zip(range(len(mags)), mags, indices):
             if abs(m - med) > 2.5 * std: # >2.5 sigma
                 outliers.append(idx)
        
        if not outliers:
             QtWidgets.QMessageBox.information(self, "Clean", "No obvious statistical outliers found.")
        else:
             # Ask user to review them
             res = QtWidgets.QMessageBox.question(self, "Outliers Found", 
                                                  f"Found {len(outliers)} frames with potential outlier magnitudes.\nReview them now?",
                                                  QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
             if res == QtWidgets.QMessageBox.Yes:
                  # Navigate to first outlier
                  self.load_frame(outliers[0])
                  self.lbl_status.setText(f"Outlier Review: Frame {outliers[0]+1} (Mag deviates significantly)")

    def prev_frame(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self.load_frame(self._current_idx)

    def next_frame(self):
        if self._current_idx < len(self.files) - 1:
            self._current_idx += 1
            self.load_frame(self._current_idx)

    def load_frame(self, idx):
        self._current_idx = idx
        self.lbl_frame.setText(f"Frame {idx+1} / {len(self.files)} : {os.path.basename(self.files[idx])}")
        
        # Load FITS
        from astropy.io import fits
        try:
            with fits.open(self.files[idx]) as hdul:
                # Assume image data is in first HDU with data
                data = None
                for hdu in hdul:
                    if hdu.data is not None:
                        data = hdu.data
                        break
                if data is not None:
                    self.current_data = data.astype('float32')
                else:
                    self.current_data = None
                
            if self.current_data is not None:
                # Optimization for display
                ny, nx = self.current_data.shape
                max_dim = 1024
                if max(ny, nx) > max_dim:
                    scale = int(np.ceil(max(ny, nx) / max_dim))
                    display_data = self.current_data[::scale, ::scale].copy()
                else:
                    display_data = self.current_data
                
                # Show on canvas with extent to keep coordinates correct
                self.canvas.show_image(display_data, extent=[0, nx, 0, ny])
                self._visualize_result()
            else:
                 self.canvas.show_image(None)
                 
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            self.current_data = None
            self.canvas.show_image(None)

    def _visualize_result(self):
        self.canvas.clear_patches()
        res = self.results_map.get(self._current_idx)
        
        if res and res.get('success'):
            xy = res.get('picked_full')
            r = res.get('aperture_used', 10)
            if xy:
                from matplotlib.patches import Circle
                # Valid detection = Green
                self.canvas.add_patch(Circle(xy, r, color='cyan', fill=False, lw=1.5))
                # Annuluses
                inner_r = r * res.get('aperture_result', {}).get('inner_coef', 2.0)
                outer_r = r * res.get('aperture_result', {}).get('outer_coef', 3.0)
                
                self.canvas.add_patch(Circle(xy, inner_r, color='yellow', fill=False, lw=1, ls='--'))
                self.canvas.add_patch(Circle(xy, outer_r, color='yellow', fill=False, lw=1, ls='--'))
                
                self.canvas.add_patch(Circle(xy, 3, color='red', fill=True))
                
                mag_val = res.get('mag')
                mag_str = f"{mag_val:.3f}" if mag_val is not None else "N/A"
                self.lbl_status.setText(f"Current: Found at ({xy[0]:.1f}, {xy[1]:.1f})\nMag: {mag_str}")
        else:
             # Not found / Failed = Red Text
             self.canvas.figure.text(0.5, 0.5, "Star Not Found", color='red', 
                                     ha='center', va='center', fontsize=20, 
                                     bbox=dict(facecolor='black', alpha=0.5))
             self.lbl_status.setText("Current: Not Found")
        
        self.canvas.draw()

    def on_canvas_click(self, x, y):
        # Manual re-measure
        if self.current_data is None: return
        
        fwhm = self.spin_fwhm.value()
        thresh = self.spin_thresh.value()
        aperture_val = self.spin_aperture.value()
        inner_coef = self.spin_inner.value()
        outer_coef = self.spin_outer.value()
        fixed = self.chk_fixed_ap.isChecked()
        
        # 1. Detect/Refine
        found_xy, params, stamp, method = pc.detect_then_refine(
            self.current_data, (x, y),
            fwhm=fwhm, threshold_sigma=thresh,
            crop_half_size=100
        )
        
        if found_xy:
            # 2. Measure
            exptime = 1.0
            old_res = self.results_map.get(self._current_idx, {})
            if old_res.get('exptime'): exptime = old_res['exptime']
            
            # Radii optimization if not fixed
            r_used = aperture_val
            radii, snrs, r_best, mag_err = None, None, aperture_val, 0.1
            
            if not fixed:
               radii, snrs, r_best, mag_err = pc.compute_snr_vs_radius(self.current_data, found_xy, fwhm=fwhm)
               r_used = r_best
            
            # Photometry
            ap_res = pc.perform_aperture_photometry(self.current_data, found_xy, r_used,
                                                   inner_coef=inner_coef, outer_coef=outer_coef,
                                                   zeropoint=0.0, exptime=exptime)
            
            # Store coefficients for visualization
            ap_res['inner_coef'] = inner_coef
            ap_res['outer_coef'] = outer_coef
            
            # Construct new result dict
            new_res = {
                'index': self._current_idx,
                'success': True,
                'msg': 'Manual Re-measure',
                'method': 'manual',
                'file': os.path.basename(self.files[self._current_idx]),
                'picked_full': found_xy,
                'radii': radii, 'snrs': snrs, 'r_best': r_best,
                'aperture_used': r_used,
                'mag': ap_res['instr_mag'],
                'mag_err': ap_res['mag_err'],
                'flux': ap_res['flux'],
                'snr': ap_res['snr'],
                'aperture_result': ap_res,
                'exptime': exptime
            }
            
            # UPDATE global map
            self.results_map[self._current_idx] = new_res
            
            # Update UI
            self._visualize_result()
            mag_val = ap_res.get('instr_mag')
            mag_str = f"{mag_val:.3f}" if mag_val is not None else "N/A"
            self.lbl_status.setText(f"Updated: Found at ({found_xy[0]:.1f}, {found_xy[1]:.1f})\nInstr Mag: {mag_str}")
            
        else:
            self.lbl_status.setText(f"Failed to find star at ({x:.1f}, {y:.1f})")
