import os
from PyQt5 import QtWidgets, QtCore
from .image_canvas import MplImageCanvas
from core import masterFrame_creator as mfc

class ReviewDialog(QtWidgets.QDialog):
    """
    Dialog to review frames in a set and mark them as bad.
    """
    def __init__(self, file_list, initial_bad=None, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Review Frames")
        self.resize(1000, 750)
        self.files = file_list
        self.bad_indices = set(initial_bad or [])
        self.current_index = 0 if file_list else -1

        self._setup_ui()
        self.show_current()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        self.canvas = MplImageCanvas(figsize=(8,6))
        layout.addWidget(self.canvas, stretch=1)
        
        self.lbl_fname = QtWidgets.QLabel("")
        self.lbl_fname.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_fname)
        
        controls = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("Previous")
        self.chk_bad = QtWidgets.QCheckBox("Bad Frame?")
        self.btn_next = QtWidgets.QPushButton("Next")
        
        controls.addWidget(self.btn_prev)
        controls.addWidget(self.chk_bad)
        controls.addWidget(self.btn_next)
        layout.addLayout(controls)
        
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch()
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_confirm = QtWidgets.QPushButton("Confirm")
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(self.btn_confirm)
        layout.addLayout(bottom)

        # Style
        self.setStyleSheet("""
            QDialog { background: #162a2a; color: #d9f0ec; }
            QPushButton { background: #0f7a73; border-radius: 4px; padding: 6px; color: white; }
            QLabel, QCheckBox { color: #d9f0ec; }
        """)

        # Connect
        self.btn_prev.clicked.connect(self.on_prev)
        self.btn_next.clicked.connect(self.on_next)
        self.chk_bad.stateChanged.connect(self.on_bad_toggled)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_confirm.clicked.connect(self.accept)

    def show_current(self):
        if self.current_index < 0 or self.current_index >= len(self.files):
            self.canvas.show_image(None)
            self.lbl_fname.setText("No files to review")
            self.chk_bad.setChecked(False)
            return
            
        path = self.files[self.current_index]
        try:
            img = mfc.load_fits(path)
            self.canvas.show_image(img)
            self.lbl_fname.setText(f"[{self.current_index+1}/{len(self.files)}] {os.path.basename(path)}")
            self.chk_bad.blockSignals(True)
            self.chk_bad.setChecked(self.current_index in self.bad_indices)
            self.chk_bad.blockSignals(False)
        except Exception as e:
            self.canvas.show_image(None)
            self.lbl_fname.setText(f"Error: {e}")

    def on_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current()

    def on_next(self):
        if self.current_index < len(self.files) - 1:
            self.current_index += 1
            self.show_current()

    def on_bad_toggled(self):
        if self.current_index < 0: return
        if self.chk_bad.isChecked():
            self.bad_indices.add(self.current_index)
        else:
            self.bad_indices.discard(self.current_index)
