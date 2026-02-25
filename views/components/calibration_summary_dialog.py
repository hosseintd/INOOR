import os
from PyQt5 import QtWidgets, QtCore, QtGui

class CalibrationSummaryDialog(QtWidgets.QDialog):
    """Professional calibration summary preview dialog with enhanced styling."""
    
    def __init__(self, summary_data, parent=None):
        super().__init__(parent)
        self.summary_data = summary_data
        self.setWindowTitle("Calibration Summary - Please Review Before Start")
        self.setModal(True)
        self.resize(900, 900)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QGroupBox {
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                font-size: 11pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 3px 0 3px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 10pt;
            }
            QScrollArea {
                background-color: #1e1e1e;
                border: none;
            }
            QPushButton {
                font-size: 10pt;
                padding: 6px;
            }
        """)
        
        self.init_ui()
        
    def create_colored_binning_text(self, binning_str):
        """Create a QTextEdit with colored binning visualization (e.g., 4096 -> 2048)."""
        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setMaximumHeight(40)
        text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 3px;
                font-family: Courier;
                font-size: 11pt;
            }
        """)
        
        # Parse binning_str like "4096x4096→2048x2048" or just "4096x4096 (no binning)"
        if "→" in binning_str or "->" in binning_str:
            parts = binning_str.replace("→", "->").split("->")
            cursor = text_edit.textCursor()
            
            # Original size in red
            fmt_red = QtGui.QTextCharFormat()
            fmt_red.setForeground(QtGui.QColor("#ff6b6b"))
            fmt_red.setFontWeight(QtGui.QFont.Bold)
            cursor.insertText(parts[0].strip(), fmt_red)
            
            # Arrow
            fmt_arrow = QtGui.QTextCharFormat()
            fmt_arrow.setForeground(QtGui.QColor("#e0e0e0"))
            cursor.insertText("  →  ", fmt_arrow)
            
            # Target size in green
            fmt_green = QtGui.QTextCharFormat()
            fmt_green.setForeground(QtGui.QColor("#51cf66"))
            fmt_green.setFontWeight(QtGui.QFont.Bold)
            cursor.insertText(parts[1].strip(), fmt_green)
        else:
            # No binning - just show in normal text
            fmt_normal = QtGui.QTextCharFormat()
            fmt_normal.setForeground(QtGui.QColor("#e0e0e0"))
            cursor = text_edit.textCursor()
            cursor.insertText(binning_str, fmt_normal)
        
        return text_edit
        
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Scrollable content
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        
        # FRAME INVENTORY
        self.add_inventory_section(scroll_layout)
        
        # PROCESSING STRATEGY
        self.add_processing_section(scroll_layout)
        
        # CALIBRATION SETTINGS
        self.add_calibration_section(scroll_layout)
        
        # LIGHT FRAME PROCESSING
        self.add_light_frame_section(scroll_layout)
        
        # OUTPUT SETTINGS
        self.add_output_section(scroll_layout)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        edit_btn = QtWidgets.QPushButton("Edit Settings")
        edit_btn.setMinimumWidth(120)
        edit_btn.clicked.connect(self.reject)  # Return to main view
        
        confirm_btn = QtWidgets.QPushButton("Confirm & Start")
        confirm_btn.setMinimumWidth(120)
        confirm_btn.setStyleSheet("background-color: #0d5e5e; font-weight: bold;")
        confirm_btn.clicked.connect(self.accept)
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setMinimumWidth(120)
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(edit_btn)
        button_layout.addStretch()
        button_layout.addWidget(confirm_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
    def create_selectable_label(self, text):
        """Create a QLineEdit that looks like a label but is selectable and copyable."""
        line_edit = QtWidgets.QLineEdit(text)
        line_edit.setReadOnly(True)
        line_edit.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 4px;
                color: #e0e0e0;
                font-size: 10pt;
            }
        """)
        return line_edit
        
    def add_inventory_section(self, parent_layout):
        grp = QtWidgets.QGroupBox("FRAME INVENTORY")
        layout = QtWidgets.QVBoxLayout()
        
        inv = self.summary_data.get('inventory', {})
        
        items = [
            ("Light Frames", inv.get('lights_summary', 'N/A')),
            ("Dark Frames", inv.get('darks_summary', 'N/A')),
            ("Flat Frames", inv.get('flats_summary', 'N/A')),
            ("Bias Frames", inv.get('bias_summary', 'N/A')),
            ("Gain Table", inv.get('gain_summary', 'None')),
            ("Hot Pixel Map", inv.get('hpm_summary', 'None')),
        ]
        
        for label, value in items:
            h_layout = QtWidgets.QHBoxLayout()
            label_widget = QtWidgets.QLabel(f"{label}:")
            label_widget.setMinimumWidth(120)
            h_layout.addWidget(label_widget)
            h_layout.addStretch()
            h_layout.addWidget(self.create_selectable_label(str(value)))
            layout.addLayout(h_layout)
        
        grp.setLayout(layout)
        parent_layout.addWidget(grp)
        
    def add_processing_section(self, parent_layout):
        grp = QtWidgets.QGroupBox("PROCESSING STRATEGY")
        layout = QtWidgets.QVBoxLayout()
        
        proc = self.summary_data.get('processing', {})
        min_shape = proc.get('min_shape', 'N/A')
        
        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(QtWidgets.QLabel("Target Resolution (Minimum Size):"))
        h_layout.addStretch()
        h_layout.addWidget(self.create_selectable_label(str(min_shape)))
        layout.addLayout(h_layout)
        layout.addSpacing(10)
        
        # Masters info
        masters = proc.get('masters', {})
        for master_name in ['dark', 'flat', 'bias']:
            if master_name in masters:
                master_info = masters[master_name]
                self.add_master_info(layout, master_name.capitalize(), master_info)
                layout.addSpacing(8)
        
        grp.setLayout(layout)
        parent_layout.addWidget(grp)
        
    def add_master_info(self, layout, name, info):
        # Master title
        title_layout = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel(f"{name} Master:")
        title_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.create_selectable_label(info.get('summary', 'N/A')))
        layout.addLayout(title_layout)
        
        # Binning info
        if info.get('binning_needed'):
            bin_layout = QtWidgets.QHBoxLayout()
            bin_layout.addSpacing(20)
            binning_text = info.get('binning_info', 'N/A')
            bin_layout.addWidget(QtWidgets.QLabel("Frame binning:"))
            bin_layout.addWidget(self.create_colored_binning_text(binning_text))
            bin_layout.addStretch()
            layout.addLayout(bin_layout)
        
        # Method info
        method_layout = QtWidgets.QHBoxLayout()
        method_layout.addSpacing(20)
        method_str = f"Method: {info.get('method', 'N/A')}"
        if info.get('sigma_clip'):
            method_str += f" | Sigma Clipping: Yes (σ_lower: {info.get('sigma_lower', 'N/A')}, σ_upper: {info.get('sigma_upper', 'N/A')})"
        method_layout.addWidget(self.create_selectable_label(method_str))
        method_layout.addStretch()
        layout.addLayout(method_layout)
        
    def add_calibration_section(self, parent_layout):
        grp = QtWidgets.QGroupBox("CALIBRATION SETTINGS")
        layout = QtWidgets.QVBoxLayout()
        
        calib = self.summary_data.get('calibration', {})
        
        items = [
            ("Dark Scaling", calib.get('dark_scaling', 'No')),
            ("Skip Bias", calib.get('skip_bias', 'No')),
            ("Auto-Flat Generation", calib.get('auto_flat', 'No')),
            ("Use GainTable", calib.get('use_gain', 'No')),
        ]
        
        for label, value in items:
            h_layout = QtWidgets.QHBoxLayout()
            label_widget = QtWidgets.QLabel(f"{label}:")
            label_widget.setMinimumWidth(160)
            h_layout.addWidget(label_widget)
            h_layout.addStretch()
            h_layout.addWidget(self.create_selectable_label(str(value)))
            layout.addLayout(h_layout)
        
        grp.setLayout(layout)
        parent_layout.addWidget(grp)
        
    def add_light_frame_section(self, parent_layout):
        grp = QtWidgets.QGroupBox("LIGHT FRAME PROCESSING")
        layout = QtWidgets.QVBoxLayout()
        
        light = self.summary_data.get('light_processing', {})
        
        # Summary
        summary_layout = QtWidgets.QHBoxLayout()
        summary_layout.addWidget(QtWidgets.QLabel("Will process:"))
        summary_layout.addStretch()
        summary_layout.addWidget(self.create_selectable_label(light.get('summary', 'N/A')))
        layout.addLayout(summary_layout)
        layout.addSpacing(8)
        
        # Binning breakdown
        binning_info = light.get('binning_breakdown', [])
        for info in binning_info:
            if "Bin to" in info:
                # Has binning - highlight it
                info_layout = QtWidgets.QHBoxLayout()
                info_layout.addSpacing(20)
                info_layout.addWidget(self.create_selectable_label(info))
                info_layout.addStretch()
            else:
                # No binning for this size
                info_layout = QtWidgets.QHBoxLayout()
                info_layout.addSpacing(20)
                info_layout.addWidget(self.create_selectable_label(info))
                info_layout.addStretch()
            layout.addLayout(info_layout)
        
        layout.addSpacing(10)
        
        # Processing sequence
        seq_label = QtWidgets.QLabel("Per-frame sequence:")
        seq_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(seq_label)
        
        sequence = light.get('sequence', [])
        for i, step in enumerate(sequence, 1):
            step_layout = QtWidgets.QHBoxLayout()
            step_layout.addSpacing(20)
            step_text = f"{i}. {step}"
            step_layout.addWidget(self.create_selectable_label(step_text))
            step_layout.addStretch()
            layout.addLayout(step_layout)
        
        grp.setLayout(layout)
        parent_layout.addWidget(grp)
        
    def add_output_section(self, parent_layout):
        grp = QtWidgets.QGroupBox("OUTPUT SETTINGS")
        layout = QtWidgets.QVBoxLayout()
        
        output = self.summary_data.get('output', {})
        
        items = [
            ("Output Binning", output.get('binning', '1x1 (None)')),
            ("Output Directory", output.get('directory', 'N/A')),
            ("Output Format", output.get('format', 'FITS 16-bit')),
            ("Preserve Original Headers", output.get('preserve_headers', 'Yes')),
            ("Add Calibration History", output.get('add_history', 'Yes')),
        ]
        
        for label, value in items:
            h_layout = QtWidgets.QHBoxLayout()
            label_widget = QtWidgets.QLabel(f"{label}:")
            label_widget.setMinimumWidth(200)
            h_layout.addWidget(label_widget)
            h_layout.addStretch()
            # Truncate long paths
            display_value = str(value)
            if label == "Output Directory" and len(display_value) > 60:
                display_value = "..." + display_value[-57:]
            h_layout.addWidget(self.create_selectable_label(display_value))
            layout.addLayout(h_layout)
        
        grp.setLayout(layout)
        parent_layout.addWidget(grp)
