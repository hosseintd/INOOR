import sys
import os
from PyQt5 import QtWidgets, QtGui

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from models.session_model import SessionModel
from views.create_master_view import CreateMasterView
from controllers.create_master_controller import CreateMasterController

from views.calibration_view import CalibrationView
from controllers.calibration_controller import CalibrationController

from models.multi_photometry_model import MultiPhotometryModel
from views.multi_photometry_view import MultiPhotometryView
from controllers.multi_photometry_controller import MultiPhotometryController

from models.astrometry_model import AstrometryModel
from views.astrometry_view import AstrometryView
from controllers.astrometry_controller import AstrometryController

from views.help_view import HelpView
from controllers.help_controller import HelpController


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iranian National Observatory Optical Reduction (INOOR) Software")
        self.resize(1400, 950)
        
        icon_path = os.path.join(current_dir, "app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

        # Tab-specific Session Models (Decoupled)
        self.session_model_master = SessionModel()
        self.session_model_calib = SessionModel()

        tabs = QtWidgets.QTabWidget()

        # --- Tab 0: Welcome / Guide 
        self.help_view = HelpView()
        self.help_controller = HelpController(self.help_view)
        tabs.addTab(self.help_view, "Welcome / Guide")

        # --- Tab 1: Create Master (New MVC) ---
        self.create_master_view = CreateMasterView()
        self.create_master_controller = CreateMasterController(self.session_model_master, self.create_master_view)
        tabs.addTab(self.create_master_view, "Create Master")

        # --- Tab 2: Calibration (New MVC) ---
        self.calibration_view = CalibrationView()
        self.calibration_controller = CalibrationController(self.session_model_calib, self.calibration_view)
        tabs.addTab(self.calibration_view, "Calibration")

        # --- Tab 3: Multi Photometry (New MVC) ---
        self.multi_phot_model = MultiPhotometryModel()
        self.multi_phot_view = MultiPhotometryView()
        self.multi_phot_controller = MultiPhotometryController(self.multi_phot_model, self.multi_phot_view)
        tabs.addTab(self.multi_phot_view, "Multi Photometry")

        # --- Tab 4: Astrometry (New MVC) ---
        self.astrometry_model = AstrometryModel()
        self.astrometry_view = AstrometryView()
        self.astrometry_controller = AstrometryController(self.astrometry_model, self.astrometry_view)
        tabs.addTab(self.astrometry_view, "Astrometry")

        self.setCentralWidget(tabs)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
