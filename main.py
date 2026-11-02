import sys
import os
from PyQt5 import QtCore, QtWidgets, QtGui

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from views.theme import ThemeSwitcher, theme_manager

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
        tabs.setDocumentMode(True)
        # Theme picker parks in the tab strip's own corner, so it is reachable
        # from every tab without spending a menu bar on it.
        tabs.setCornerWidget(ThemeSwitcher(), QtCore.Qt.TopRightCorner)

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

    def closeEvent(self, event):
        """Stop every running worker before the window goes away.

        Without this, quitting during a long job destroyed a Python object that
        still owned a running QThread and Qt aborted the process.
        """
        from utils.thread_guard import stop_workers

        total = 0
        for name in ('create_master_controller', 'calibration_controller',
                     'multi_phot_controller', 'astrometry_controller',
                     'help_controller'):
            ctrl = getattr(self, name, None)
            if ctrl is not None:
                total += stop_workers(ctrl)
        if total:
            from utils.logging_config import get_logger
            get_logger(__name__).info("Stopped %d running worker(s) on exit", total)
        super().closeEvent(event)

def _install_excepthook():
    """Log and show unhandled exceptions instead of letting Qt abort silently.

    PyQt5 calls abort() on an exception that escapes a slot, so a bug in a
    handler took the whole application down with nothing on screen and, in a
    frozen build, nothing anywhere. Now it is logged to Output/logs/inoor.log
    and shown, and the app stays up so unsaved work is not lost.
    """
    import traceback
    from utils.logging_config import get_logger

    log = get_logger("inoor")
    previous = sys.excepthook

    def handler(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc, tb)
            return
        log.error("Unhandled exception", exc_info=(exc_type, exc, tb))
        try:
            QtWidgets.QMessageBox.critical(
                None, "Unexpected Error",
                f"{exc_type.__name__}: {exc}\n\n"
                "The application is still running, but this operation did not "
                "complete. Details are in Output/logs/inoor.log.\n\n"
                + "".join(traceback.format_exception(exc_type, exc, tb))[-1500:])
        except Exception:
            previous(exc_type, exc, tb)

    sys.excepthook = handler


def main():
    from utils.logging_config import setup_logging
    setup_logging()
    _install_excepthook()

    # Must be set before the QApplication exists, or Qt ignores them and the
    # whole UI renders at 1x on a scaled 4K display.
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    # Organisation/app names are what QSettings keys off, so set them before
    # anything reads a saved preference.
    app.setOrganizationName("INO")
    app.setApplicationName("INOOR")
    theme_manager.apply(theme_manager.load_saved(), save=False)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
