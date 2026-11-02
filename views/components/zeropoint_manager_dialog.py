"""
Inspect and edit the zeropoints already stored for each frame.

Applying a calibration writes one zeropoint per frame into the model, and from
then on every reported magnitude depends on it. Without a way to look at those
numbers afterwards there is no way to notice that one frame was calibrated off a
star that had drifted, or that half the sequence was calibrated with a different
aperture rule than the other half. This dialog exposes them: what the value is,
which reference star produced it, and which aperture rule was in force.
"""
import os
from PyQt5 import QtCore, QtWidgets

from views.theme import set_variant


class ZeropointManagerDialog(QtWidgets.QDialog):
    def __init__(self, parent, files, zeropoints, zeropoint_meta):
        super().__init__(parent)
        self.setWindowTitle("Stored zeropoints")
        self.resize(940, 560)
        self._files = list(files)
        self._zp = dict(zeropoints or {})
        self._meta = dict(zeropoint_meta or {})
        self._cleared_all = False
        # Styling comes from the application stylesheet (views/theme.py).
        self._build()
        self.populate()

    def _build(self):
        root = QtWidgets.QVBoxLayout(self)

        self.lbl_info = QtWidgets.QLabel()
        self.lbl_info.setWordWrap(True)
        root.addWidget(self.lbl_info)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Frame", "Zeropoint (editable)", "Reference star(s)", "Aperture rule", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        root.addWidget(self.table)

        btns = QtWidgets.QHBoxLayout()
        self.btn_clear_sel = QtWidgets.QPushButton("Clear selected")
        self.btn_clear_all = QtWidgets.QPushButton("Clear ALL zeropoints")
        set_variant(self.btn_clear_all, "danger")
        self.btn_clear_all.setToolTip(
            "Remove every stored zeropoint. Magnitudes go back to instrumental "
            "(zeropoint 0) until you calibrate again.")
        self.btn_ok = QtWidgets.QPushButton("Apply changes")
        set_variant(self.btn_ok, "primary")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        btns.addWidget(self.btn_clear_sel)
        btns.addWidget(self.btn_clear_all)
        btns.addStretch()
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        root.addLayout(btns)

        self.btn_clear_sel.clicked.connect(self.on_clear_selected)
        self.btn_clear_all.clicked.connect(self.on_clear_all)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def populate(self):
        self.table.setRowCount(len(self._files))
        n_set = 0
        for row, path in enumerate(self._files):
            fname = os.path.basename(path)
            has = fname in self._zp
            n_set += 1 if has else 0

            it = QtWidgets.QTableWidgetItem(fname)
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, it)

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.01)
            spin.setSpecialValueText("")   # blank when nothing is stored
            spin.setValue(float(self._zp.get(fname, 0.0)))
            spin.setEnabled(has)
            self.table.setCellWidget(row, 1, spin)

            meta = self._meta.get(fname) or {}
            refs = meta.get("refs") or []
            if refs:
                txt = "; ".join(
                    f"{r.get('name', 'ref')} m={r.get('ref_mag', float('nan')):.3f} "
                    f"@({r.get('x', float('nan')):.0f},{r.get('y', float('nan')):.0f}) "
                    f"ZP={r.get('zp', float('nan')):.3f}"
                    for r in refs)
            else:
                txt = "-" if has else ""
            it2 = QtWidgets.QTableWidgetItem(txt)
            it2.setFlags(it2.flags() & ~QtCore.Qt.ItemIsEditable)
            it2.setToolTip(txt)
            self.table.setItem(row, 2, it2)

            it3 = QtWidgets.QTableWidgetItem(str(meta.get("policy", "") or ("-" if has else "")))
            it3.setFlags(it3.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 3, it3)

            it4 = QtWidgets.QTableWidgetItem("stored" if has else "not calibrated")
            it4.setFlags(it4.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 4, it4)

        self.lbl_info.setText(
            f"<b>{n_set}</b> of <b>{len(self._files)}</b> frames have a stored zeropoint. "
            "Edit a value to override it, or clear it so that frame falls back to "
            "instrumental magnitudes.")

    def on_clear_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            QtWidgets.QMessageBox.information(self, "Nothing selected",
                                              "Select one or more rows first.")
            return
        for r in rows:
            fname = self.table.item(r, 0).text()
            self._zp.pop(fname, None)
            self._meta.pop(fname, None)
        self.populate()

    def on_clear_all(self):
        if QtWidgets.QMessageBox.question(
                self, "Clear all zeropoints?",
                "Remove the stored zeropoint from every frame?\n\n"
                "Reported magnitudes go back to instrumental until you calibrate "
                "again. Measured fluxes and instrumental magnitudes are not affected.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        self._zp.clear()
        self._meta.clear()
        self._cleared_all = True
        self.populate()

    def results(self):
        """(zeropoints, meta, cleared_all) after the user's edits."""
        out = {}
        for r in range(self.table.rowCount()):
            fname = self.table.item(r, 0).text()
            w = self.table.cellWidget(r, 1)
            if w is not None and w.isEnabled() and fname in self._zp:
                out[fname] = float(w.value())
        return out, dict(self._meta), self._cleared_all
