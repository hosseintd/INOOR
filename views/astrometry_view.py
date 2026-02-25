# views/astrometry_view.py

import os
from PyQt5 import QtWidgets, QtCore, QtGui
from .components.astrometry.image_viewer import AstrometryFitsViewer
from .components.astrometry.control_panel import AstrometryControlPanel

class AstrometryView(QtWidgets.QWidget):
    # Signals for Controller
    open_fits_clicked = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # --- LEFT PANEL: Viewer ---
        self.viewer = AstrometryFitsViewer()
        layout.addWidget(self.viewer, stretch=4)

        # --- RIGHT PANEL: Controls ---
        self.right_container = QtWidgets.QFrame()
        self.right_container.setFixedWidth(360)
        self.right_layout = QtWidgets.QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(10)

        # Control Panel
        self.control_panel = AstrometryControlPanel()
        self.right_layout.addWidget(self.control_panel)

        # Catalog/Info Group
        self.grp_info = QtWidgets.QGroupBox("Source Info & Catalog")
        self.info_layout = QtWidgets.QVBoxLayout(self.grp_info)
        
        self.info_box = QtWidgets.QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setPlaceholderText("Click a marker to see details...")
        self.info_box.setFixedHeight(140)
        self.info_layout.addWidget(self.info_box)

        self.btn_query_ps1 = QtWidgets.QPushButton("Query PS1 for Selected Star")
        self.btn_export_csv = QtWidgets.QPushButton("Export Annotated Stars (CSV)")
        self.info_layout.addWidget(self.btn_query_ps1)
        self.info_layout.addWidget(self.btn_export_csv)
        
        self.right_layout.addWidget(self.grp_info)

        # Transient/Other (Future expansion)
        self.btn_open_fits = QtWidgets.QPushButton("Open New FITS Image")
        self.btn_open_fits.setMinimumHeight(35)
        self.btn_open_fits.setStyleSheet("background-color: #0f7a73; font-weight: bold;")
        self.right_layout.addWidget(self.btn_open_fits)

        self.right_layout.addStretch()
        
        layout.addWidget(self.right_container, stretch=1)

        # Styling (Match dark teal theme with high-contrast elements)
        self.setStyleSheet("""
            QWidget { background: #162a2a; color: #e0fcf9; font-family: 'Segoe UI', Arial; font-size: 10pt; }
            QGroupBox { border: 2px solid #2e6f6f; margin-top: 12px; font-weight: bold; border-radius: 5px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; color: #3df2e5; }
            QPushButton { background: #0f7a73; border: 1px solid #2e6f6f; border-radius: 4px; padding: 8px; color: white; font-weight: bold; }
            QPushButton:hover { background: #14a098; }
            QPushButton:pressed { background: #0a524d; }
            QTextEdit { background: #0a0f0f; color: #e0fcf9; border: 1px solid #2e6f6f; font-family: 'Consolas', 'Courier New', monospace; line-height: 1.4; }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit { background: #0e1a1a; color: #e0fcf9; border: 1px solid #2e6f6f; padding: 4px; border-radius: 3px; }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus { border: 1px solid #3df2e5; }
            QCheckBox { color: #e0fcf9; spacing: 8px; }
            QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #2e6f6f; background: #0e1a1a; }
            QCheckBox::indicator:checked { background: #3df2e5; }
            QLabel { color: #e0fcf9; }
        """)

        # Connect internal signal
        self.btn_open_fits.clicked.connect(self.open_fits_clicked.emit)
        self.viewer.set_info_widget(self.info_box)

    def update_status(self, msg):
        # We could add a status label if needed
        pass
