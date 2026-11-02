import os
from PyQt5 import QtWidgets, QtCore, QtGui

from .theme import set_variant, theme_manager

class HelpView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.zoom_factor = 1.0
        self.text_labels = []
        self.image_labels = []
        self.section_widgets = {}
        self.init_ui()
        self._apply_theme()
        theme_manager.changed.connect(self._apply_theme)

    def _apply_theme(self, *_):
        """Colours that cannot come from the global stylesheet.

        The credits block is rich text with inline colours, and the sidebar has
        its own selected/hover treatment, so both are rebuilt per theme.
        """
        t = theme_manager.tokens()
        self.sidebar_panel.setStyleSheet(
            f"background-color: {t['bg_surface']};"
            f"border-right: 1px solid {t['border_muted']};")
        self.sidebar.setStyleSheet(f"""
            QListWidget#helpSidebar {{
                background-color: transparent;
                color: {t['text']};
                border: none;
                font-size: 11pt;
                outline: none;
                padding-top: 10px;
            }}
            QListWidget#helpSidebar::item {{
                padding: 14px 18px;
                border-bottom: 1px solid {t['border_muted']};
            }}
            QListWidget#helpSidebar::item:selected {{
                background-color: {t['accent']};
                color: {t['text_on_accent']};
                font-weight: bold;
            }}
            QListWidget#helpSidebar::item:hover:!selected {{
                background-color: {t['bg_raised']};
            }}
        """)
        self.zoom_widget.setStyleSheet(
            f"background-color: {t['bg_base']};"
            f"border-top: 1px solid {t['border_muted']};")
        self.content_widget.setStyleSheet(f"background-color: {t['bg_base']};")
        self.scroll_area.setStyleSheet(
            f"QScrollArea {{ background-color: {t['bg_base']}; border: none; }}")

        for lbl in self.heading_labels:
            lbl.setStyleSheet(
                f"color: {t['text_heading']}; border-bottom: 2px solid {t['border']};")
        for lbl in self.image_frames:
            lbl.setStyleSheet(
                f"border: 1px solid {t['border_muted']}; border-radius: 4px;")
        self.lbl_credits.setText(self._credits_html(t))
        self._fit_paragraph(self.lbl_credits)

    def _fit_paragraph(self, lbl):
        """The width of the images, and the height the text needs at that width.

        The scroll area squeezes the layout to its minimum size, and a word-wrapped
        label's minimum is shorter than its text, so a long paragraph lost its end.
        """
        width = int(800 * self.zoom_factor)
        lbl.ensurePolished()
        lbl.setFixedWidth(width)
        lbl.setMinimumHeight(lbl.heightForWidth(width))

    def init_ui(self):
        # Main layout
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left Sidebar Panel
        # Colours are applied in _apply_theme(); this only builds the widgets.
        self.sidebar_panel = sidebar_panel = QtWidgets.QWidget()
        sidebar_panel.setFixedWidth(200)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_panel)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Left Sidebar for navigation
        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setObjectName("helpSidebar")

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
        self.zoom_widget = zoom_widget = QtWidgets.QWidget()
        zoom_layout = QtWidgets.QHBoxLayout(zoom_widget)
        zoom_layout.setContentsMargins(10, 12, 10, 12)

        self.btn_zoom_out = QtWidgets.QPushButton("−")
        self.btn_zoom_out.setCursor(QtCore.Qt.PointingHandCursor)
        set_variant(self.btn_zoom_out, "primary")

        self.btn_zoom_in = QtWidgets.QPushButton("+")
        self.btn_zoom_in.setCursor(QtCore.Qt.PointingHandCursor)
        set_variant(self.btn_zoom_in, "primary")

        self.lbl_zoom = QtWidgets.QLabel("Zoom: 100%")
        self.lbl_zoom.setAlignment(QtCore.Qt.AlignCenter)
        set_variant(self.lbl_zoom, "muted")

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

        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(30, 0, 30, 40)
        self.content_layout.setSpacing(20)
        self.content_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)

        self.help_dir = os.path.join(os.path.dirname(__file__), "..", "utils", "resources", "help")
        self.build_document()

        self.scroll_area.setWidget(self.content_widget)

        main_layout.addWidget(sidebar_panel)
        main_layout.addWidget(self.scroll_area)

    @staticmethod
    def _credits_html(t):
        """Rich text carries its own link colour, so it is rebuilt per theme."""
        link = t["accent_hover"] if t["is_dark"] else t["accent"]
        a = f'style="color: {link}; text-decoration: none;"'
        return f"""
        <p style='line-height:1.6;'>
        <b>Version:</b> V1.0.0 <br><br>
        <b>Developer:</b> <a href="https://github.com/hosseintd" {a}>Hossein Torkzadeh</a> and Soroush Shakeri (Advisor)<br>
        <b>Repository:</b> <a href="https://github.com/hosseintd/INOOR" {a}>GitHub / INOOR</a><br>
        <b>License:</b> GNU General Public License v3<br><br>
        Special thanks to Parisa Hashemi, Saeid Karimi Mianrudi, Hamed Altafi and Reza Rezaei for their invaluable comments and fruitful discussions.<br>
        We also thank the INO340 Scientific and Technical staff.<br><br>
        <i>Developed for the Iranian National Observatory (INO340).</i>
        </p>
        """

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
        # Collected so _apply_theme can recolour them; they are not reachable
        # through a plain class selector because each needs its own rule.
        self.heading_labels = []
        self.image_frames = []

        # 1. Native Banner
        self.lbl_banner = QtWidgets.QLabel()
        self.lbl_banner.setAlignment(QtCore.Qt.AlignCenter)
        self.movie = QtGui.QMovie(os.path.join(self.help_dir, "images", "welcome_banner.webp"))
        if self.movie.isValid():
            self.lbl_banner.setMovie(self.movie)
            self.movie.start()
        
        self.lbl_title = QtWidgets.QLabel("INOOR")
        self.lbl_title.setAlignment(QtCore.Qt.AlignCenter)
        # Had no colour at all, so it fell back to the palette default (black)
        # and was invisible against the dark background. "brand" sets the
        # colour without a font-size, leaving the zoom buttons in control.
        set_variant(self.lbl_title, "brand")


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
                "text": "Stacking raw calibration frames such as Bias, Dark, and Flat is the first step of the reduction. The Create Master tab combines several frames of one type into a single master frame. You can choose between Median and Mean stacking methods. The Median method is recommended as it effectively removes cosmic ray strikes and passing satellite trails (e.g., in sky flats). Flats can be scaled to a common level before they are combined, because the sky fades during a sequence of sky flats; 'Scale frames to a common level' is ticked for you when every file name contains the word flat. Bias and Dark frames must not be scaled.\n\nYou can also apply Sigma Clipping, a filter that runs after combining and replaces outlier pixels with the local median. It is off by default, because on a master bias or dark it also erases the stable hot pixels that the master has to subtract. Use 'Bad Frame?' to leave a frame out. 'Start Creating' shows a preview of the master, and the master is saved once you confirm it.\n\nInteractive Histogram Tools are provided on the right panel to help you evaluate your raw data. Drag the vertical green and red lines to adjust the display range."
            },
            {
                "id": "calibration",
                "title": "2. Calibration Pipeline",
                "image": "tab_calibration.png",
                "text": "The Calibration tab turns raw Light frames into calibrated frames. Load each set with 'Type to Load' and 'Browse and Load Files'. Light and Dark frames are required; Bias, Flat and a HotPixelMap are optional. A set can be raw frames, which are combined with the options you choose, or one master frame from the Create Master tab. Select a set or a frame to look at it with 'Review Selected (Large)' or to remove it with 'Remove Selected Files/Set'. 'Check Flat Frames' flags flats that should not go into the master. Then press 'START CALIBRATION' at the bottom and choose the output folder; each Light is saved with the prefix Calibrated_.\n\nIf the loaded sets have mismatched dimensions, the application performs Auto Binning to resample them to the smallest common resolution. A master dark already contains the bias, so subtracting it removes the dark current and the bias together, and no master bias is needed when the exposure times match. If they differ, tick 'Scale Dark by Exposure Time': the master bias is taken out of the master dark, the remaining dark current is scaled to the exposure of each Light, and the bias is put back (we suggest using it only when you are sure the sensor is in the linear regime).\n\nFlat fielding then corrects vignetting and dust shadows using the master flat. If no Flat frames are loaded, INOOR asks whether to make an auto-flat: a low-order polynomial fitted to the sky of each Light with the stars masked, by which that Light is divided. The auto-flat corrects only the large-scale pattern, and not the dust shadows or the pixel-to-pixel response, so it suits a quick reduction. If you answer No, the calibration does not start.\n\nFinally, optional steps clean and bin the result. 'Remove Cosmic Rays / Hot Pixels' replaces a pixel that deviates from the median of a small box by more than the chosen threshold. A HotPixelMap replaces each bad pixel by the mean of the good pixels in a 3x3 box, enlarged if needed. 'Output Binning' averages the pixels before the frame is saved.",
                "flowchart": "calibration_flowchart.png"
            },
            {
                "id": "photometry",
                "title": "3. Multi Photometry",
                "image": "tab_photometry.png",
                "text": "After the frames are calibrated, use the Multi Photometry tab to measure instrumental and apparent magnitudes. Add the calibrated frames with the 'Add' button and click a star on the image. INOOR looks for the star inside the 'Detection Stamp' around the click, then shows its magnitude, error and SNR, with its radial profile and its SNR curve. The gain, read noise and dark charge of the noise model follow the camera mode (High, Low or Merge) and the binning that it reads from the header of each frame.\n\nThe aperture radius follows one of four rules: 'Frame FWHM x factor' (the default), where every star in a frame shares one radius; 'Max SNR (per star)'; 'Curve of growth (per star)'; and a fixed radius ('Fixed aperture for all'). Use the shared radius for light curves and whenever stars are compared, as in a zeropoint. The sky level comes from an annulus around the aperture.\n\n'BULK PHOTOMETRY' measures the selected star in every loaded frame. Its position in each frame is predicted from the stars of the field and refined once with a centroid. A star near saturation, or with a PSF broader than that of the frame, is flagged in the summary, in the result panel and in the CSV file.\n\n'Calculate Zeropoints' finds the zeropoint from reference stars of known magnitude. 'Calc Extinction (k)' fits the atmospheric extinction coefficient over a range of airmass. 'Review Results Table' and 'Show Light Curve' display the results, and 'Save CSV' writes them to a file."
            },
            {
                "id": "astrometry",
                "title": "4. Astrometry & Catalog",
                "image": "tab_astrometry.png",
                "text": "The Astrometry tab is used for Plate Solving, which links the image pixels to Right Ascension and Declination. Load your image with the 'Open New FITS Image' button and enter your Astrometry.net API key (the ? button next to the field explains how to get one).\n\n'SOLVE PLATE' uploads the image to nova.astrometry.net, so an internet connection and the key are needed. 'Downsample' reduces the image before the upload. The frame is saved with its solution in the Output/Astrometry folder, and the path is shown when the solve ends. The center and the size of the field (width x height) appear below the image.\n\nAfter the solve, the green circles are the reference stars of astrometry.net and, with 'Show detected markers' on, the red circles are the sources it detected in the frame. Click a circle to see its pixel position, RA and Dec in the right pane. 'Query PS1 for Selected Star' then fetches the g, r, i, z and y magnitudes of that star from the Pan-STARRS1 catalog at MAST. 'Export Annotated Stars (CSV)' writes the detected sources with their positions and PS1 magnitudes to a file."
            }
        ]

        for s in SECTIONS:
            lbl_heading = QtWidgets.QLabel(s["title"])
            self.heading_labels.append(lbl_heading)
            self.content_layout.addWidget(lbl_heading)
            self.text_labels.append(("h2", lbl_heading, 24, True))

            if s["id"] != "introduction":
                self.section_widgets[s["id"]] = lbl_heading

            if "image" in s:
                lbl_img = QtWidgets.QLabel()
                lbl_img.setAlignment(QtCore.Qt.AlignCenter)
                self.image_frames.append(lbl_img)
                path = os.path.join(self.help_dir, "images", s["image"])
                pixmap = QtGui.QPixmap(path)
                self.image_labels.append((lbl_img, pixmap, 800))
                self.content_layout.addWidget(lbl_img)

            lbl_txt = QtWidgets.QLabel(s["text"].replace('\n', '<br>'))
            lbl_txt.setWordWrap(True)
            lbl_txt.setContentsMargins(0, 0, 0, 20)
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
        self.heading_labels.append(lbl_credits_head)
        self.content_layout.addWidget(lbl_credits_head)
        self.text_labels.append(("h2", lbl_credits_head, 24, True))
        
        self.section_widgets["about"] = lbl_credits_head

        self.lbl_credits = QtWidgets.QLabel()
        self.lbl_credits.setWordWrap(True)
        self.lbl_credits.setOpenExternalLinks(True)
        self.content_layout.addWidget(self.lbl_credits)
        self.text_labels.append(("p", self.lbl_credits, 15, False))

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
            if typ == "p":
                self._fit_paragraph(lbl)

        # Scale explicit images cleanly (SmoothTransformation preserves maximum detail of original images)
        for lbl, pixmap, base_width in self.image_labels:
            if not pixmap.isNull():
                scaled_w = int(base_width * self.zoom_factor)
                lbl.setPixmap(pixmap.scaledToWidth(scaled_w, QtCore.Qt.SmoothTransformation))
