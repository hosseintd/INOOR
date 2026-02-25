from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PhotometryTableDialog(QtWidgets.QDialog):
    def __init__(self, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Photometry Results Table - Review")
        self.resize(1100, 700)
        self.rows = rows # List of PhotometryRow objects or dicts
        self.updated_rows = list(rows) 
        
        self.cols = ["index", "filename", "x", "y", "mag", "mag_err", "flux", "snr", "zeropoint", "date_ut", "date_obs"]
        self._setup_ui()
        self.populate()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.setStyleSheet("""
            QDialog { background: #162a2a; color: #d9f0ec; }
            QTableWidget { background: #101616; color: #d9f0ec; gridline-color: #2e6f6f; }
            QHeaderView::section { background: #2e6f6f; color: #d9f0ec; padding: 4px; border: 1px solid #0f7a73; }
            QPushButton { background: #0f7a73; color: white; border-radius: 4px; padding: 6px; }
        """)
        
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(len(self.cols) + 1) # +1 for checkbox
        self.table.setHorizontalHeaderLabels([""] + [c.capitalize() for c in self.cols])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.table)
        
        btns = QtWidgets.QHBoxLayout()
        self.btn_remove = QtWidgets.QPushButton("Remove Selected")
        self.btn_save_csv = QtWidgets.QPushButton("Save CSV")
        self.btn_save_zp = QtWidgets.QPushButton("Save Zeropoints")
        self.btn_load_zp = QtWidgets.QPushButton("Load Zeropoints")
        self.btn_plot = QtWidgets.QPushButton("Light Curve")
        self.btn_close = QtWidgets.QPushButton("Close")
        
        btns.addWidget(self.btn_remove)
        btns.addWidget(self.btn_save_csv)
        btns.addWidget(self.btn_save_zp)
        btns.addWidget(self.btn_load_zp)
        btns.addWidget(self.btn_plot)
        btns.addStretch()
        btns.addWidget(self.btn_close)
        layout.addLayout(btns)
        
        self.btn_remove.clicked.connect(self.on_remove)
        self.btn_save_csv.clicked.connect(self.on_save_csv)
        self.btn_save_zp.clicked.connect(self.on_save_zp)
        self.btn_load_zp.clicked.connect(self.on_load_zp)
        self.btn_plot.clicked.connect(self.on_plot)
        self.btn_close.clicked.connect(self.accept)

    def populate(self):
        self.table.setRowCount(len(self.updated_rows))
        for i, r in enumerate(self.updated_rows):
            # Checkbox
            ck = QtWidgets.QCheckBox()
            ck.setStyleSheet("margin-left:5px;")
            self.table.setCellWidget(i, 0, ck)
            
            # Data
            # r can be a dataclass or a dict
            def get_val(obj, key):
                if isinstance(obj, dict): return obj.get(key)
                return getattr(obj, key, None)

            for j, col in enumerate(self.cols):
                val = get_val(r, col)
                txt = "N/A" if val is None else (f"{val:.4f}" if isinstance(val, float) else str(val))
                item = QtWidgets.QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(i, j+1, item)
        
        self.table.setColumnWidth(0, 30)
        self.table.resizeColumnsToContents()

    def on_remove(self):
        to_del = []
        for i in range(self.table.rowCount()):
            if self.table.cellWidget(i, 0).isChecked():
                to_del.append(i)
        
        if not to_del: return
        
        to_del.sort(reverse=True)
        for idx in to_del:
            self.table.removeRow(idx)
            self.updated_rows.pop(idx)

    def on_save_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "photometry.csv", "CSV (*.csv)")
        if not path: return
        
        data = []
        for r in self.updated_rows:
            d = {}
            for c in self.cols:
                if hasattr(r, c): d[c] = getattr(r, c)
                elif isinstance(r, dict): d[c] = r.get(c)
            data.append(d)
        
        pd.DataFrame(data).to_csv(path, index=False)
        QtWidgets.QMessageBox.information(self, "Saved", f"Table saved to {path}")

    def on_save_zp(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Zeropoints", "zeropoints.csv", "CSV (*.csv)")
        if not path: return
        
        zp_data = []
        for r in self.updated_rows:
            fname = getattr(r, 'filename', r.get('filename') if isinstance(r, dict) else None)
            zp = getattr(r, 'zeropoint', r.get('zeropoint', 0.0) if isinstance(r, dict) else 0.0)
            if fname:
                zp_data.append({'filename': fname, 'zeropoint': zp})
        
        pd.DataFrame(zp_data).to_csv(path, index=False)

    def on_load_zp(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Zeropoints", "", "CSV (*.csv)")
        if not path: return
        try:
            df = pd.read_csv(path)
            # Match by filename
            # ... implementation ...
            QtWidgets.QMessageBox.information(self, "Note", "Loading ZP from CSV is handled by the main controller via signal/callback.")
        except: pass

    def on_plot(self):
        # Quick plot
        mags = []
        for r in self.updated_rows:
            m = getattr(r, 'mag', r.get('mag') if isinstance(r, dict) else None)
            if m is not None: mags.append(m)
        
        if not mags: return
        plt.figure("Quick Lightcurve")
        plt.plot(mags, 'o-')
        plt.gca().invert_yaxis()
        plt.show()
