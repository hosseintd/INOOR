# views/astrometry_view.py

import os
from PyQt5 import QtWidgets, QtCore, QtGui
from .components.astrometry.image_viewer import AstrometryFitsViewer
from .components.astrometry.control_panel import AstrometryControlPanel
from .theme import set_variant

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
        # "&&" escapes the ampersand. With a single "&", Qt reads "&C" as a
        # mnemonic and the title rendered as "Source Info  Catalog".
        self.grp_info = QtWidgets.QGroupBox("Source Info && Catalog")
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
        set_variant(self.btn_open_fits, "primary")
        self.right_layout.addWidget(self.btn_open_fits)

        self.right_layout.addStretch()

        layout.addWidget(self.right_container, stretch=1)

        # The info box is the one place a monospace font earns its keep --
        # coordinates and magnitudes line up column-wise.
        self.info_box.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.FixedFont))

        # Everything else comes from the application stylesheet (views/theme.py).

        # Connect internal signal
        self.btn_open_fits.clicked.connect(self.open_fits_clicked.emit)
        self.viewer.set_info_widget(self.info_box)

    def update_status(self, msg):
        # We could add a status label if needed
        pass
