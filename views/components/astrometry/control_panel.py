# views/components/astrometry/control_panel.py
from PyQt5.QtWidgets import QWidget, QFormLayout, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLabel, QHBoxLayout, QVBoxLayout, QCheckBox, QGroupBox, QMessageBox, QTextBrowser, QDialog
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QDesktopServices
from utils.astrometry_config import SCALE_PRESETS, ASTROMETRY_TIMEOUT, ASTROMETRY_API_KEY
from utils.gui_helpers import CollapsibleBox

class AstrometryControlPanel(QWidget):
    # Signals for Controller
    solve_clicked = pyqtSignal(dict) # params
    preset_changed = pyqtSignal(str)
    stretch_changed = pyqtSignal(str, float, float)
    view_action = pyqtSignal(str) # 'rotate_cw', 'rotate_ccw', 'flip_h', 'flip_v', 'toggle_det'
    cmap_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Solver Settings Group
        grp_solve = QGroupBox("Astrometry.net Settings")
        form = QFormLayout(grp_solve)
        
        # API Key with help button
        api_key_layout = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("Enter your Astrometry.net API key")
        if ASTROMETRY_API_KEY:
            self.api_key_edit.setText(ASTROMETRY_API_KEY)
        self.btn_api_help = QPushButton('?')
        self.btn_api_help.setMaximumWidth(30)
        self.btn_api_help.setToolTip('Click for API key help')
        self.btn_api_help.clicked.connect(self._show_api_help)
        api_key_layout.addWidget(self.api_key_edit)
        api_key_layout.addWidget(self.btn_api_help)
        form.addRow('API Key:', api_key_layout)

        self.scale_combo = QComboBox()
        presets = list(SCALE_PRESETS.keys())
        self.scale_combo.addItems(presets)
        if 'tiny' in presets:
            self.scale_combo.setCurrentText('tiny')
        form.addRow('Scale Preset:', self.scale_combo)

        self.scale_low = QDoubleSpinBox()
        self.scale_low.setSuffix(' arcmin'); self.scale_low.setRange(0.001, 1e6)
        self.scale_high = QDoubleSpinBox()
        self.scale_high.setSuffix(' arcmin'); self.scale_high.setRange(0.001, 1e6)
        
        initial_preset = self.scale_combo.currentText()
        self.scale_low.setValue(SCALE_PRESETS[initial_preset][0])
        self.scale_high.setValue(SCALE_PRESETS[initial_preset][1])
        form.addRow('Scale Lower:', self.scale_low)
        form.addRow('Scale Upper:', self.scale_high)

        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 16); self.downsample_spin.setValue(4)
        form.addRow('Downsample:', self.downsample_spin)
        
        # --- Advanced Settings (Collapsible) ---
        self.adv_box = CollapsibleBox("Advanced Settings")
        adv_layout = QFormLayout()
        adv_layout.setContentsMargins(4, 4, 4, 4)
        
        # Timeout
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(30, 86400); self.timeout_spin.setValue(ASTROMETRY_TIMEOUT)
        adv_layout.addRow('Timeout (s):', self.timeout_spin)
        
        # Enable/Disable RA/Dec hint
        self.chk_use_radec = QCheckBox("Use RA/Dec hint for faster solving")
        adv_layout.addRow(self.chk_use_radec)
        
        # Optional RA/Dec for faster solving
        ra_layout = QHBoxLayout()
        self.ra_spin = QDoubleSpinBox()
        self.ra_spin.setSuffix('°'); self.ra_spin.setRange(0.0, 360.0); self.ra_spin.setDecimals(6)
        self.ra_spin.setEnabled(False)
        self.btn_ra_converter = QPushButton('HMS→°')
        self.btn_ra_converter.setMaximumWidth(65)
        self.btn_ra_converter.setEnabled(False)
        self.btn_ra_converter.clicked.connect(self._show_ra_converter)
        ra_layout.addWidget(self.ra_spin)
        ra_layout.addWidget(self.btn_ra_converter)
        adv_layout.addRow('RA:', ra_layout)
        
        dec_layout = QHBoxLayout()
        self.dec_spin = QDoubleSpinBox()
        self.dec_spin.setSuffix('°'); self.dec_spin.setRange(-90.0, 90.0); self.dec_spin.setDecimals(6)
        self.dec_spin.setEnabled(False)
        self.btn_dec_converter = QPushButton('DMS→°')
        self.btn_dec_converter.setMaximumWidth(65)
        self.btn_dec_converter.setEnabled(False)
        self.btn_dec_converter.clicked.connect(self._show_dec_converter)
        dec_layout.addWidget(self.dec_spin)
        dec_layout.addWidget(self.btn_dec_converter)
        adv_layout.addRow('Dec:', dec_layout)
        
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setSuffix(' arcmin'); self.radius_spin.setRange(1.0, 10800.0); self.radius_spin.setValue(300.0); self.radius_spin.setDecimals(1)
        self.radius_spin.setEnabled(False)
        adv_layout.addRow('Search Radius:', self.radius_spin)
        
        self.adv_box.set_content_layout(adv_layout)
        form.addRow(self.adv_box)

        self.btn_solve = QPushButton('SOLVE PLATE')
        self.btn_solve.setMinimumHeight(40)
        self.btn_solve.setStyleSheet("background-color: #008b8b; color: #ffffff; border: 2px solid #3df2e5; font-size: 11pt;")
        form.addRow(self.btn_solve)
        
        layout.addWidget(grp_solve)

        # Viewer Controls Group
        grp_view = QGroupBox("Viewer Controls")
        v_layout = QVBoxLayout(grp_view)
        
        rot_row = QHBoxLayout()
        self.btn_rot_ccw = QPushButton('Rotate -90°')
        self.btn_rot_cw = QPushButton('Rotate +90°')
        rot_row.addWidget(self.btn_rot_ccw); rot_row.addWidget(self.btn_rot_cw)
        v_layout.addLayout(rot_row)

        flip_row = QHBoxLayout()
        self.btn_flip_h = QPushButton('Flip H')
        self.btn_flip_v = QPushButton('Flip V')
        flip_row.addWidget(self.btn_flip_h); flip_row.addWidget(self.btn_flip_v)
        v_layout.addLayout(flip_row)

        self.chk_show_det = QCheckBox('Show detected markers')
        self.chk_show_det.setChecked(False)
        v_layout.addWidget(self.chk_show_det)

        form_v = QFormLayout()
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['gray','gray_r','viridis','plasma','inferno','magma'])
        form_v.addRow('Colormap:', self.cmap_combo)

        self.stretch_combo = QComboBox()
        self.stretch_combo.addItems(['zscale','minmax'])
        form_v.addRow('Stretch:', self.stretch_combo)
        
        v_layout.addLayout(form_v)
        layout.addWidget(grp_view)
        layout.addStretch()

        # Connections
        self.btn_solve.clicked.connect(self._on_solve)
        self.chk_use_radec.toggled.connect(self._on_radec_toggled)
        self.scale_combo.currentTextChanged.connect(self.preset_changed.emit)
        self.btn_rot_ccw.clicked.connect(lambda: self.view_action.emit('rotate_ccw'))
        self.btn_rot_cw.clicked.connect(lambda: self.view_action.emit('rotate_cw'))
        self.btn_flip_h.clicked.connect(lambda: self.view_action.emit('flip_h'))
        self.btn_flip_v.clicked.connect(lambda: self.view_action.emit('flip_v'))
        self.chk_show_det.stateChanged.connect(lambda: self.view_action.emit('toggle_det'))
        self.cmap_combo.currentTextChanged.connect(self.cmap_changed.emit)
        self.stretch_combo.currentTextChanged.connect(lambda t: self.stretch_changed.emit(t, 0, 0))

    def _on_solve(self):
        api_key = self.api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(None, "API Key Required", "Please enter your Astrometry.net API key.\n\nDon't have one? Click the '?' button for instructions.")
            return
        
        # Get RA/Dec if enabled
        ra = None
        dec = None
        if self.chk_use_radec.isChecked():
            ra = self.ra_spin.value()
            dec = self.dec_spin.value()
        
        params = {
            'api_key': api_key,
            'scale_low': self.scale_low.value(),
            'scale_high': self.scale_high.value(),
            'timeout': self.timeout_spin.value(),
            'downsample': self.downsample_spin.value(),
            'ra': ra,
            'dec': dec,
            'search_radius': self.radius_spin.value() if (ra and dec) else None
        }
        self.solve_clicked.emit(params)
    
    def _on_radec_toggled(self, checked):
        """Enable/disable RA/Dec input fields based on checkbox state."""
        self.ra_spin.setEnabled(checked)
        self.dec_spin.setEnabled(checked)
        self.radius_spin.setEnabled(checked)
        self.btn_ra_converter.setEnabled(checked)
        self.btn_dec_converter.setEnabled(checked)
        # Note: Values are NOT reset, user's inputs are preserved
    
    def _show_api_help(self):
        """Show API help dialog with clickable link."""
        dialog = QDialog(None)
        dialog.setWindowTitle("How to Get an API Key")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        
        # Use QTextBrowser for clickable links
        text_browser = QTextBrowser()
        text_browser.setMarkdown("""### To get an Astrometry.net API key:

1. **Visit the API Help page**: [https://nova.astrometry.net/api_help](https://nova.astrometry.net/api_help)

2. **Create an account** (or sign in if you have one)

3. **Go to your profile settings** and find your API key

4. **Copy your API key** from the settings page

5. **Paste it** in the API Key field above

---

The API key is **required** to use the plate solving service.

Need help? Visit the link above for more information.""")
        text_browser.setOpenExternalLinks(True)
        layout.addWidget(text_browser)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def _show_ra_converter(self):
        """Show RA converter dialog (HMS to decimal degrees)."""
        dialog = QDialog(None)
        dialog.setWindowTitle("RA Converter (HMS → Degrees)")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        
        # Instruction
        layout.addWidget(QLabel("Enter RA in Hours:Minutes:Seconds"))
        
        # Input fields
        form_layout = QFormLayout()
        hours_spin = QSpinBox()
        hours_spin.setRange(0, 23)
        minutes_spin = QSpinBox()
        minutes_spin.setRange(0, 59)
        seconds_spin = QDoubleSpinBox()
        seconds_spin.setRange(0, 59.999)
        seconds_spin.setDecimals(3)
        
        form_layout.addRow("Hours:", hours_spin)
        form_layout.addRow("Minutes:", minutes_spin)
        form_layout.addRow("Seconds:", seconds_spin)
        layout.addLayout(form_layout)
        
        # Result display
        result_label = QLabel("Result: 0.0°")
        result_label.setStyleSheet("font-weight: bold; color: #3df2e5;")
        
        def update_result():
            h = hours_spin.value()
            m = minutes_spin.value()
            s = seconds_spin.value()
            degrees = (h + m/60.0 + s/3600.0) * 15.0  # RA: 1 hour = 15 degrees
            result_label.setText(f"Result: {degrees:.6f}°")
        
        hours_spin.valueChanged.connect(update_result)
        minutes_spin.valueChanged.connect(update_result)
        seconds_spin.valueChanged.connect(update_result)
        update_result()
        
        layout.addWidget(result_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        convert_btn = QPushButton("Use This Value")
        cancel_btn = QPushButton("Cancel")
        
        def on_convert():
            h = hours_spin.value()
            m = minutes_spin.value()
            s = seconds_spin.value()
            degrees = (h + m/60.0 + s/3600.0) * 15.0
            self.ra_spin.setValue(degrees)
            dialog.accept()
        
        convert_btn.clicked.connect(on_convert)
        cancel_btn.clicked.connect(dialog.reject)
        
        btn_layout.addWidget(convert_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec_()
    
    def _show_dec_converter(self):
        """Show Dec converter dialog (DMS to decimal degrees)."""
        dialog = QDialog(None)
        dialog.setWindowTitle("Dec Converter (DMS → Degrees)")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        
        # Instruction
        layout.addWidget(QLabel("Enter Dec in Degrees:Minutes:Seconds"))
        
        # Input fields
        form_layout = QFormLayout()
        sign_combo = QComboBox()
        sign_combo.addItems(["+", "-"])
        degrees_spin = QSpinBox()
        degrees_spin.setRange(0, 90)
        minutes_spin = QSpinBox()
        minutes_spin.setRange(0, 59)
        seconds_spin = QDoubleSpinBox()
        seconds_spin.setRange(0, 59.999)
        seconds_spin.setDecimals(3)
        
        form_layout.addRow("Sign:", sign_combo)
        form_layout.addRow("Degrees:", degrees_spin)
        form_layout.addRow("Minutes:", minutes_spin)
        form_layout.addRow("Seconds:", seconds_spin)
        layout.addLayout(form_layout)
        
        # Result display
        result_label = QLabel("Result: 0.0°")
        result_label.setStyleSheet("font-weight: bold; color: #3df2e5;")
        
        def update_result():
            sign = 1 if sign_combo.currentText() == "+" else -1
            d = degrees_spin.value()
            m = minutes_spin.value()
            s = seconds_spin.value()
            degrees = sign * (d + m/60.0 + s/3600.0)
            result_label.setText(f"Result: {degrees:.6f}°")
        
        sign_combo.currentTextChanged.connect(update_result)
        degrees_spin.valueChanged.connect(update_result)
        minutes_spin.valueChanged.connect(update_result)
        seconds_spin.valueChanged.connect(update_result)
        update_result()
        
        layout.addWidget(result_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        convert_btn = QPushButton("Use This Value")
        cancel_btn = QPushButton("Cancel")
        
        def on_convert():
            sign = 1 if sign_combo.currentText() == "+" else -1
            d = degrees_spin.value()
            m = minutes_spin.value()
            s = seconds_spin.value()
            degrees = sign * (d + m/60.0 + s/3600.0)
            self.dec_spin.setValue(degrees)
            dialog.accept()
        
        convert_btn.clicked.connect(on_convert)
        cancel_btn.clicked.connect(dialog.reject)
        
        btn_layout.addWidget(convert_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec_()

    def set_scales(self, low, high):
        self.scale_low.setValue(low)
        self.scale_high.setValue(high)
