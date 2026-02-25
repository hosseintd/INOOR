import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pandas as pd

class ZeropointCalibrationDialog(QtWidgets.QDialog):
    # Signals to request actions from Controller
    request_add_ref = QtCore.pyqtSignal()
    request_review = QtCore.pyqtSignal(int) # index of column to review

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
        self.setStyleSheet("""
            QDialog { background: #162a2a; color: #d9f0ec; }
            QTableWidget { background: #101616; color: #d9f0ec; }
            QHeaderView::section { background: #2e6f6f; color: #d9f0ec; }
            QPushButton { background: #0f7a73; color: white; border-radius: 4px; padding: 6px; }
            QLabel { color: #d9f0ec; font-weight: bold; }
            QDoubleSpinBox { background: #0e1a1a; color: #d9f0ec; border: 1px solid #2e6f6f; }
            QGroupBox { border: 1px solid #2e6f6f; margin-top: 10px; }
        """)
        layout = QtWidgets.QVBoxLayout(self)
        
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
        
        self.btn_add_star.clicked.connect(self.request_add_ref.emit)
        self.btn_recac.clicked.connect(self.on_calculate)
        self.btn_review.clicked.connect(self.on_review_click)
        self.btn_remove_col.clicked.connect(self.on_remove_col)
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        
        if not self.history:
            self.btn_apply.setEnabled(False)
            self.btn_recac.setEnabled(False)

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
