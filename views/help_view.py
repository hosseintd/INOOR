import os
from PyQt5 import QtWidgets, QtCore, QtGui

class HelpView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.zoom_factor = 1.0
        self.text_labels = []
        self.image_labels = []
        self.section_widgets = {}
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left Sidebar Panel
        sidebar_panel = QtWidgets.QWidget()
        sidebar_panel.setFixedWidth(200)
        sidebar_panel.setStyleSheet("background-color: #1a2e2e; border-right: 1px solid #2e6f6f;")
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_panel)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Left Sidebar for navigation
        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setObjectName("helpSidebar")
        self.sidebar.setStyleSheet("""
            QListWidget#helpSidebar {
                background-color: transparent;
                color: #d9f0ec;
                border: none;
                font-size: 14px;
                outline: none;
                padding-top: 10px;
            }
            QListWidget#helpSidebar::item {
                padding: 15px 20px;
                border-bottom: 1px solid #233f3f;
            }
            QListWidget#helpSidebar::item:selected {
                background-color: #0f7a73;
                color: white;
                font-weight: bold;
            }
            QListWidget#helpSidebar::item:hover:!selected {
                background-color: #1c3d3d;
            }
        """)

        categories = [
            "Welcome",
            "Create Master",
            "Calibration",
            "Multi Photometry",
            "Astrometry",
            "About"
        ]
        self.sidebar.addItems(categories)
        self.sidebar.setCurrentRow(0)

        # Zoom Controls
        zoom_widget = QtWidgets.QWidget()
        zoom_widget.setStyleSheet("background-color: #162a2a; border-top: 1px solid #2e6f6f;")
        zoom_layout = QtWidgets.QHBoxLayout(zoom_widget)
        zoom_layout.setContentsMargins(10, 15, 10, 15)
        
        btn_style = """
            QPushButton {
                background-color: #0f7a73;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover { background-color: #139c93; }
        """
        
        self.btn_zoom_out = QtWidgets.QPushButton("-")
        self.btn_zoom_out.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_zoom_out.setStyleSheet(btn_style)
        
        self.btn_zoom_in = QtWidgets.QPushButton("+")
        self.btn_zoom_in.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_zoom_in.setStyleSheet(btn_style)
        
        self.lbl_zoom = QtWidgets.QLabel("Zoom: 100%")
        self.lbl_zoom.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_zoom.setStyleSheet("color: #d9f0ec; font-size: 13px; border: none;")

        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self.btn_zoom_in.clicked.connect(self._zoom_in)

        zoom_layout.addWidget(self.btn_zoom_out)
        zoom_layout.addWidget(self.lbl_zoom)
        zoom_layout.addWidget(self.btn_zoom_in)

        sidebar_layout.addWidget(self.sidebar)
        sidebar_layout.addWidget(zoom_widget)

        # Right Content Area (QScrollArea)
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { background-color: #162a2a; border: none; } QScrollBar:vertical { background: #162a2a; width: 10px; } QScrollBar::handle:vertical { background: #2e6f6f; border-radius: 5px; } QScrollBar::handle:vertical:hover { background: #0f7a73; }")

        self.content_widget = QtWidgets.QWidget()
        self.content_widget.setStyleSheet("background-color: #162a2a;")
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(30, 0, 30, 40)
        self.content_layout.setSpacing(20)
        self.content_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)

        self.help_dir = os.path.join(os.path.dirname(__file__), "..", "utils", "resources", "help")
        self.build_document()

        self.scroll_area.setWidget(self.content_widget)

        main_layout.addWidget(sidebar_panel)
        main_layout.addWidget(self.scroll_area)

    def load_help_file(self, file_path):
        pass # Not used in native layout mode

    def scroll_to_anchor(self, anchor):
        if anchor in self.section_widgets:
            QtCore.QTimer.singleShot(50, lambda: self.scroll_area.ensureWidgetVisible(self.section_widgets[anchor], 0, 50))

    def _zoom_in(self):
        if self.zoom_factor < 2.5:
            self.zoom_factor += 0.1
            self.update_zoom()

    def _zoom_out(self):
        if self.zoom_factor > 0.5:
            self.zoom_factor -= 0.1
            self.update_zoom()

    def build_document(self):
        # 1. Native Banner
        self.lbl_banner = QtWidgets.QLabel()
        self.lbl_banner.setAlignment(QtCore.Qt.AlignCenter)
        self.movie = QtGui.QMovie(os.path.join(self.help_dir, "images", "welcome_banner.webp"))
        if self.movie.isValid():
            self.lbl_banner.setMovie(self.movie)
            self.movie.start()
        
        self.lbl_title = QtWidgets.QLabel("INOOR")
        self.lbl_title.setAlignment(QtCore.Qt.AlignCenter)
        
        self.content_layout.addWidget(self.lbl_banner)
        self.content_layout.addWidget(self.lbl_title)
        
        self.section_widgets["introduction"] = self.lbl_title
        self.text_labels.append(("title", self.lbl_title, 36, True))

        # 2. SECTIONS
        SECTIONS = [
            {
                "id": "introduction",
                "title": "Getting Started",
                "text": "Welcome to the <b>Iranian National Observatory Optical Reduction (INOOR)</b> software. This application processes raw astronomical data from the telescope through a complete pipeline to final photometry and astrometry. If this is your first time using the software, follow the tabs from left to right to process your dataset."
            },
            {
                "id": "master",
                "title": "1. Create Master Frames",
                "image": "tab_master.png",
                "text": "Stacking raw calibration frames such as Bias, Dark, and Flat is the first step in reducing electrical and thermal noise. The Create Master tab allows you to combine multiple identical frames into a single master frame. You can choose between Median and Mean stacking methods. The Median method is recommended as it effectively removes cosmic ray strikes and passing satellite trails (e.g., in sky flats). You can also apply Sigma Clipping, which identifies and rejects outlier pixels after combining.\n\nInteractive Histogram Tools are provided on the right panel to help you evaluate the optical quality of your raw data. Drag the vertical green and red sliders to adjust the display range and visually inspect noise characteristics."
            },
            {
                "id": "calibration",
                "title": "2. Calibration Pipeline",
                "image": "tab_calibration.png",
                "text": "The Calibration tab automates the dark subtraction and flat fielding stages. It handles multi-binning geometry and scales varying exposure times. When you configure the tables and click the 'START CALIBRATION' button, the software begins processing the selected light frames. After loading calibration files via the left panel buttons, you can review individual frames or remove corrupted exposures using the 'Review Selected' and 'Remove Selected' buttons. Once the list is ready, press the 'START CALIBRATION' button at the bottom to create and save the master frame.\n\nIf the loaded frame batches have mismatched dimensions, the application performs Auto Binning to resample them to the smallest common resolution. Thermal noise is scaled by exposure time if exact matching dark frames are not available (but we suggest using it only when you are sure the sensor is in the linear regime). Standard Bias subtraction removes the sensor readout bias using the Master Bias frame. Field Flattening then corrects vignetting and dust shadows using the Master Flat. Finally, an optional Cosmic Ray and Hot Pixel filtering pass uses median spatial filters and a 3x3 interpolation to clean possible cosmic rays and hot/bad pixels.",
                "flowchart": "calibration_flowchart.png"
            },
            {
                "id": "photometry",
                "title": "3. Multi Photometry",
                "image": "tab_photometry.png",
                "text": "After the frames are calibrated, use the Multi Photometry tab to extract instrumental and apparent magnitudes from the selected targets. This tab is designed for tracking stellar brightness across sequential frames.\n\nLoad the dataset of light frames using the designated buttons on the left sidebar. The central panel processes Aperture Extraction, which operates as an automated flux measurement for the chosen objects across the viewing duration. The application can calculate the atmospheric extinction coefficient from the data points to account for localized airmass changes. You can update the Zero-Point alignment by observing known reference catalog stars in the frame to determine absolute magnitudes. When the generation sequence is complete, you can review the plotted interactive light curves directly on the panel interface."
            },
            {
                "id": "astrometry",
                "title": "4. Astrometry & Catalog",
                "image": "tab_astrometry.png",
                "text": "The Astrometry tab is used for Plate Solving, which transforms arbitrary image pixel coordinates into standard Right Ascension and Declination celestial coordinates. To begin, load your image with the Open FITS image button.\n\nExecuting Plate Solving on the frame analyzes the star field to calculate the exact center orientation and rotation angle. You can then trigger the Catalog Overlay interaction, which fetches and matches the visual field stars against the Pan-STARRS1 reference catalog. Once the target is successfully plate-solved, the Point-and-Click spatial query feature becomes active. Select any specific target object in the image array to instantly view its designation name, photometric magnitude (by clicking on the Query button), and celestial coordinates in the right reporting pane."
            }
        ]

        for s in SECTIONS:
            lbl_heading = QtWidgets.QLabel(s["title"])
            lbl_heading.setStyleSheet("color: #aaffff; border-bottom: 2px solid #2e6f6f;")
            self.content_layout.addWidget(lbl_heading)
            self.text_labels.append(("h2", lbl_heading, 24, True))

            if s["id"] != "introduction":
                self.section_widgets[s["id"]] = lbl_heading

            if "image" in s:
                lbl_img = QtWidgets.QLabel()
                lbl_img.setAlignment(QtCore.Qt.AlignCenter)
                lbl_img.setStyleSheet("border: 1px solid #2e6f6f; border-radius: 4px;")
                path = os.path.join(self.help_dir, "images", s["image"])
                pixmap = QtGui.QPixmap(path)
                self.image_labels.append((lbl_img, pixmap, 800))
                self.content_layout.addWidget(lbl_img)

            lbl_txt = QtWidgets.QLabel(s["text"].replace('\n', '<br>'))
            lbl_txt.setWordWrap(True)
            lbl_txt.setStyleSheet("color: #ecf0f1; margin-bottom: 20px;")
            self.content_layout.addWidget(lbl_txt)
            self.text_labels.append(("p", lbl_txt, 15, False))

            if "flowchart" in s:
                lbl_flow = QtWidgets.QLabel()
                lbl_flow.setAlignment(QtCore.Qt.AlignCenter)
                path = os.path.join(self.help_dir, "images", s["flowchart"])
                pixmap = QtGui.QPixmap(path)
                self.image_labels.append((lbl_flow, pixmap, 800))
                self.content_layout.addWidget(lbl_flow)

        # 3. Credits Section
        self.content_layout.addSpacing(40)
        lbl_credits_head = QtWidgets.QLabel("About")
        lbl_credits_head.setStyleSheet("color: #aaffff; border-bottom: 2px solid #2e6f6f;")
        self.content_layout.addWidget(lbl_credits_head)
        self.text_labels.append(("h2", lbl_credits_head, 24, True))
        
        self.section_widgets["about"] = lbl_credits_head

        credits_html = """
        <p style='line-height:1.6;'>
        <b>Version:</b> V1.0.0 <br><br>
        <b>Developer:</b> <a href="https://github.com/hosseintd" style="color: #66ccff; text-decoration: none;">Hossein Torkzadeh</a><br>
        <b>Repository:</b> <a href="https://github.com/hosseintd/INOOR" style="color: #66ccff; text-decoration: none;">GitHub / INOOR</a><br><br>
        <b>IUT Team Members:</b> Saeid Karimi and Parisa Hashemi<br>
        <b>Supervisor:</b> Dr. Soroush Shakeri<br>
        <b>Special Thanks To:</b> Hamed Altafi and Dr. Reza Rezaei, for their invaluable recommendations.<br><br>
        <b>Institution:</b> Isfahan University of Technology (IUT)<br>
        <b>Affiliation:</b> Iranian National Observatory (INO)
        </p>
        """
        lbl_credits = QtWidgets.QLabel(credits_html)
        lbl_credits.setWordWrap(True)
        lbl_credits.setOpenExternalLinks(True)
        lbl_credits.setStyleSheet("color: #ecf0f1;")
        self.content_layout.addWidget(lbl_credits)
        self.text_labels.append(("p", lbl_credits, 15, False))

        # 4. Logos Horizontal Layout
        logo_layout = QtWidgets.QHBoxLayout()
        logo_layout.setAlignment(QtCore.Qt.AlignCenter)
        logo_layout.setSpacing(40)
        
        lbl_ino = QtWidgets.QLabel()
        lbl_ino.setAlignment(QtCore.Qt.AlignCenter)
        pixmap_ino = QtGui.QPixmap(os.path.join(self.help_dir, "images", "INO-Logo.png"))
        self.image_labels.append((lbl_ino, pixmap_ino, 220))

        lbl_iut = QtWidgets.QLabel()
        lbl_iut.setAlignment(QtCore.Qt.AlignCenter)
        pixmap_iut = QtGui.QPixmap(os.path.join(self.help_dir, "images", "IUT-Logo.png"))
        self.image_labels.append((lbl_iut, pixmap_iut, 160))

        logo_layout.addWidget(lbl_ino)
        logo_layout.addWidget(lbl_iut)
        self.content_layout.addLayout(logo_layout)

        # Apply initial zoom sizing
        self.update_zoom()

    def update_zoom(self):
        self.lbl_zoom.setText(f"Zoom: {int(self.zoom_factor * 100)}%")
        
        # Scale banner
        if self.movie.isValid():
            self.movie.setScaledSize(QtCore.QSize(int(800 * self.zoom_factor), int(533 * self.zoom_factor)))

        # Scale fonts
        for typ, lbl, base_size, is_bold in self.text_labels:
            font = lbl.font()
            if typ == "title":
                font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, int(8 * self.zoom_factor))
            font.setPointSize(max(8, int(base_size * self.zoom_factor)))
            font.setBold(is_bold)
            lbl.setFont(font)

        # Scale explicit images cleanly (SmoothTransformation preserves maximum detail of original images)
        for lbl, pixmap, base_width in self.image_labels:
            if not pixmap.isNull():
                scaled_w = int(base_width * self.zoom_factor)
                lbl.setPixmap(pixmap.scaledToWidth(scaled_w, QtCore.Qt.SmoothTransformation))
