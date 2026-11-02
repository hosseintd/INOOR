"""
One QApplication for the whole test session, so that a run does not depend on the order of the files.

A test module's fixture that creates the application also releases it when the module ends, and the
module-level theme manager of views/theme.py does not survive that: every later test that builds a
widget then fails with "wrapped C/C++ object of type _ThemeManager has been deleted". The program itself
has one QApplication for its whole life.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
