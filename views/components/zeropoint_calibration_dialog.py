import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pandas as pd

class ZeropointCalibrationDialog(QtWidgets.QDialog):
    # Signals to request actions from Controller
    request_add_ref = QtCore.pyqtSignal()
    request_review = QtCore.pyqtSignal(int) # index of column to review
    # (policy, parameter) -- re-measure every reference star under a new rule
    request_repolicy = QtCore.pyqtSignal(str, float)

    def __init__(self, parent, files, calibration_history):
        """
        calibration_history: List of dicts, each representing a "run":
          {
             'ref_mag': float,
             'results': dict (index -> result_dict),
             'name': str (optional)
          }
        """
        super().__init__(parent)
        self.setWindowTitle("Zeropoint Calibration")
        self.resize(1100, 600)
        self.files = files
        self.history = calibration_history
        self.zeropoints = {} # file_index -> final_zp
        
        self.spin_mags = [] # Keep track of spinboxes for each column
        
        self._setup_ui()
        self.populate()

    def _setup_ui(self):
        # Styling comes from the application stylesheet (views/theme.py).
        layout = QtWidgets.QVBoxLayout(self)
        
        # Aperture rule. A zeropoint is only transferable to another star if
        # both were measured through the same radius, so the rule that produced
        # these numbers belongs on this screen, not buried in Advanced Settings.
        rule_row = QtWidgets.QHBoxLayout()
        rule_row.addWidget(QtWidgets.QLabel("Aperture rule:"))
        self.cmb_rule = QtWidgets.QComboBox()
        self.cmb_rule.addItem("Max SNR (per star)", "snr")
        self.cmb_rule.addItem("Curve of growth (per star)", "growth")
        self.cmb_rule.addItem("Frame FWHM x factor", "frame_fwhm")
        self.cmb_rule.setToolTip(
            "Frame FWHM gives every star in a frame the same radius, which is\n"
            "what makes a zeropoint measured on one star valid for another.")
        self.spin_rule_param = QtWidgets.QDoubleSpinBox()
        self.spin_rule_param.setRange(0.05, 20.0); self.spin_rule_param.setDecimals(2)
        self.spin_rule_param.setSingleStep(0.05); self.spin_rule_param.setValue(1.0)
        self.btn_apply_rule = QtWidgets.QPushButton("Re-measure with this rule")
        rule_row.addWidget(self.cmb_rule)
        rule_row.addWidget(self.spin_rule_param)
        rule_row.addWidget(self.btn_apply_rule)
        rule_row.addStretch()
        layout.addLayout(rule_row)

        # Mags Panel
        self.mags_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(self.mags_layout)
        
        # Table
        self.table = QtWidgets.QTableWidget()
        layout.addWidget(self.table)
        
        # Buttons
        btns = QtWidgets.QHBoxLayout()
        self.btn_add_star = QtWidgets.QPushButton("Add Ref Star")
        self.btn_recac = QtWidgets.QPushButton("Recalculate ZP")
        self.btn_review = QtWidgets.QPushButton("Review ZPs (Visual)")
        self.btn_remove_col = QtWidgets.QPushButton("Remove Last Star")
        
        self.btn_apply = QtWidgets.QPushButton("Apply Zeropoints")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        
        btns.addWidget(self.btn_add_star)
        btns.addWidget(self.btn_recac)
        btns.addWidget(self.btn_review)
        btns.addWidget(self.btn_remove_col)
        btns.addStretch()
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)
        
        self.cmb_rule.currentIndexChanged.connect(self._sync_rule_param)
        self.btn_apply_rule.clicked.connect(
            lambda: self.request_repolicy.emit(self.cmb_rule.currentData() or 'snr',
                                               float(self.spin_rule_param.value())))
        self._sync_rule_param()
        self.btn_add_star.clicked.connect(self.request_add_ref.emit)
        self.btn_recac.clicked.connect(self.on_calculate)
        self.btn_review.clicked.connect(self.on_review_click)
        self.btn_remove_col.clicked.connect(self.on_remove_col)
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        
        if not self.history:
            self.btn_apply.setEnabled(False)
            self.btn_recac.setEnabled(False)

    def set_rule(self, policy, param):
        """Show the rule the controller is actually using."""
        i = self.cmb_rule.findData(policy)
        if i >= 0:
            self.cmb_rule.blockSignals(True)
            self.cmb_rule.setCurrentIndex(i)
            self.cmb_rule.blockSignals(False)
        self._sync_rule_param()
        self.spin_rule_param.setValue(float(param))

    def _sync_rule_param(self):
        pol = self.cmb_rule.currentData() or 'snr'
        if pol == 'frame_fwhm':
            self.spin_rule_param.setRange(0.3, 6.0); self.spin_rule_param.setSingleStep(0.1)
            self.spin_rule_param.setPrefix("x ")
        else:
            self.spin_rule_param.setRange(0.50, 0.995); self.spin_rule_param.setSingleStep(0.01)
            self.spin_rule_param.setPrefix("E ")
        self.spin_rule_param.setVisible(pol != 'snr')

    def populate(self):
        import os
        # Header setup
        # Cols: File, [Instr1, ZP1] ... [InstrN, ZPN], Mean ZP
        self.cols = ["File"]
        for i in range(len(self.history)):
            self.cols.extend([f"Instr_{i+1}", f"ZP_{i+1}"])
        self.cols.append("Mean ZP")
        
        self.table.setColumnCount(len(self.cols))
        self.table.setHorizontalHeaderLabels(self.cols)
        
        # Refresh Mags UI
        # Clear layout items safely
        while self.mags_layout.count():
            child = self.mags_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        self.spin_mags = []
        for i, run in enumerate(self.history):
            g = QtWidgets.QGroupBox(f"Ref Star {i+1}")
            gl = QtWidgets.QVBoxLayout(g)
            lbl = QtWidgets.QLabel("App Mag:")
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-30, 30); sp.setDecimals(3); sp.setValue(run['ref_mag'])
            gl.addWidget(lbl); gl.addWidget(sp)
            self.mags_layout.addWidget(g)
            self.spin_mags.append(sp)
            
        self.mags_layout.addStretch()
            
        # Cleared, not merged into. populate() only ever wrote into this dict,
        # so removing a reference star left its zeropoint behind: the cell showed
        # "N/A" while get_results() still handed the stale number to the model.
        self.zeropoints = {}

        self.table.setRowCount(len(self.files))
        for r, path in enumerate(self.files):
            fname = os.path.basename(path)
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(fname))
            
            zps_for_row = []
            
            col_idx = 1
            for i, run in enumerate(self.history):
                res = run['results'].get(r, {})
                # Get raw instrumental mag
                ap_res = res.get('aperture_result', {})
                raw_instr = ap_res.get('instr_mag')
                
                # Instr Item
                txt_instr = f"{raw_instr:.4f}" if raw_instr is not None else "N/A"
                self.table.setItem(r, col_idx, QtWidgets.QTableWidgetItem(txt_instr))
                
                # ZP Item
                if raw_instr is not None:
                    zp = run['ref_mag'] - raw_instr
                    zps_for_row.append(zp)
                    txt_zp = f"{zp:.4f}"
                else:
                    txt_zp = "N/A"
                self.table.setItem(r, col_idx+1, QtWidgets.QTableWidgetItem(txt_zp))
                
                col_idx += 2
                
            # Mean ZP
            if zps_for_row:
                mean_zp = np.mean(zps_for_row)
                self.zeropoints[r] = mean_zp
                self.table.setItem(r, col_idx, QtWidgets.QTableWidgetItem(f"{mean_zp:.4f}"))
            else:
                self.table.setItem(r, col_idx, QtWidgets.QTableWidgetItem("N/A"))
                
        self.table.resizeColumnsToContents()

    def on_calculate(self):
        # Update mags in history
        for i, sp in enumerate(self.spin_mags):
            self.history[i]['ref_mag'] = sp.value()
        # Repopulate calls populate which re-calcs ZP
        self.populate()

    def on_remove_col(self):
        if self.history:
            self.history.pop()
            self.populate()
            
    def on_review_click(self):
        if not self.history: return
        # If multiple, ask which one
        idx = 0
        if len(self.history) > 1:
            opts = [f"Ref Star {i+1}" for i in range(len(self.history))]
            item, ok = QtWidgets.QInputDialog.getItem(self, "Select Reference", "Review which star?", opts, 0, False)
            if not ok: return
            idx = opts.index(item)
        
        self.request_review.emit(idx)

    def get_results(self):
        return self.zeropoints
