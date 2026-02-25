from PyQt5 import QtWidgets, QtCore

class DisplaySettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent, current_max_dim):
        super().__init__(parent)
        self.setWindowTitle("Display Optimization Settings")
        self.resize(350, 200)
        
        self.current_max_dim = current_max_dim
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog { background: #162a2a; color: #d9f0ec; }
            QLabel { color: #d9f0ec; font-weight: bold; }
            QSpinBox { background: #0e1a1a; color: #d9f0ec; border: 1px solid #2e6f6f; padding: 4px; }
            QPushButton { background: #0f7a73; color: white; border-radius: 4px; padding: 6px; }
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        info = QtWidgets.QLabel("Tune visualization resolution to optimize speed.\n"
                                "Higher values show more detail but are slower.\n"
                                "Scientific measurements are NOT affected.")
        info.setWordWrap(True)
        info.setStyleSheet("font-weight: normal; font-style: italic; color: #aaffff; margin-bottom: 10px;")
        layout.addWidget(info)
        
        form = QtWidgets.QFormLayout()
        self.spin_dim = QtWidgets.QSpinBox()
        self.spin_dim.setRange(512, 8192)
        self.spin_dim.setSingleStep(256)
        self.spin_dim.setValue(self.current_max_dim)
        
        form.addRow("Max Display Dimension (px):", self.spin_dim)
        layout.addLayout(form)
        
        btns = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton("Apply && Refresh")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        btns.addStretch()
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)
        
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def get_value(self):
        return self.spin_dim.value()
