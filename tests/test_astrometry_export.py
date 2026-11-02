"""
"Export Annotated Stars (CSV)" in the Astrometry tab. No network: the PS1 cone search is a stub.
"""
import csv

import pytest
import requests
from astropy.table import Table

import controllers.astrometry_workers as aw
import core.astrometry.catalog_query as cq


def _ps1_stub(ra, dec, radius_deg, **kw):
    return Table({'objID': [1], 'raMean': [ra], 'decMean': [dec], 'nDetections': [20],
                  'gMeanPSFMag': [17.805], 'rMeanPSFMag': [17.305], 'iMeanPSFMag': [17.090],
                  'zMeanPSFMag': [16.993], 'yMeanPSFMag': [16.902]})


def test_export_keeps_the_pixel_y_next_to_the_ps1_y_band(tmp_path, monkeypatch):
    """The PS1 y-band magnitude used to overwrite the pixel y (both keys were 'y')."""
    monkeypatch.setattr(aw, 'ps1cone', _ps1_stub)
    out = tmp_path / 'export.csv'
    stars = [{'kind': 'det', 'x': 3050.91, 'y': 2228.70, 'ra_deg': 333.8781304, 'dec_deg': 51.6041528}]
    aw.BatchExportWorker(stars, str(out)).run()

    with open(out, newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == ['x', 'y', 'ra_deg', 'dec_deg',
                             'g_ps1', 'r_ps1', 'i_ps1', 'z_ps1', 'y_ps1']
    assert float(rows[0]['y']) == 2228.70
    assert float(rows[0]['y_ps1']) == 16.902
    assert float(rows[0]['g_ps1']) == 17.805


def _offline(*a, **kw):
    raise ConnectionError("Failed to establish a new connection")


def test_failed_queries_are_recorded_not_swallowed(tmp_path, monkeypatch):
    monkeypatch.setattr(aw, 'ps1cone', _offline)
    stars = [{'kind': 'det', 'x': 1.0, 'y': 2.0, 'ra_deg': 333.88, 'dec_deg': 51.60},
             {'kind': 'det', 'x': 3.0, 'y': 4.0, 'ra_deg': 333.89, 'dec_deg': 51.61}]
    w = aw.BatchExportWorker(stars, str(tmp_path / 'export.csv'))
    w.run()
    assert len(w.failed_queries) == 2
    assert 'Failed to establish a new connection' in w.failed_queries[0]


def test_export_with_failed_queries_is_not_reported_as_success(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import controllers.astrometry_controller as ac
    from models.astrometry_model import AstrometryModel
    from views.astrometry_view import AstrometryView

    boxes = []
    for kind in ('information', 'warning', 'critical'):
        monkeypatch.setattr(ac.QMessageBox, kind,
                            staticmethod(lambda *a, _k=kind: boxes.append((_k, a[1], a[2]))))
    monkeypatch.setattr(aw, 'ps1cone', _offline)
    ctrl = ac.AstrometryController(AstrometryModel(), AstrometryView())
    out = str(tmp_path / 'export.csv')
    ctrl.exp_worker = aw.BatchExportWorker([{'kind': 'det', 'x': 1.0, 'y': 2.0,
                                             'ra_deg': 333.88, 'dec_deg': 51.60}], out)
    ctrl.exp_prog = QtWidgets.QProgressDialog()
    ctrl.exp_worker.run()
    ctrl.on_export_finished(out)

    assert [b[:2] for b in boxes] == [('warning', 'Export Incomplete')]
    assert '1 of 1 stars' in boxes[0][2]


def test_only_reference_stars_means_nothing_to_export(monkeypatch):
    """The detected sources are the red markers; the green ones are astrometry.net's reference stars."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import controllers.astrometry_controller as ac
    from models.astrometry_model import AstrometryModel
    from views.astrometry_view import AstrometryView

    texts = []
    monkeypatch.setattr(ac.QMessageBox, 'information',
                        staticmethod(lambda parent, title, text: texts.append(text)))
    view = AstrometryView()
    monkeypatch.setattr(view.viewer, 'export_annotated_list',
                        lambda: [{'kind': 'ref', 'x': 1.0, 'y': 2.0, 'ra_deg': 333.88, 'dec_deg': 51.60}])
    ac.AstrometryController(AstrometryModel(), view).on_export_csv()
    assert texts == ["No detected stars to export."]


def _query_errors():
    worker = aw.CatalogQueryWorker(333.88, 51.60)
    errors, tables = [], []
    worker.error.connect(errors.append)
    worker.finished.connect(tables.append)
    worker.run()
    assert tables == []
    return errors


@pytest.mark.parametrize("cause", [requests.exceptions.ConnectionError("Max retries exceeded"),
                                   requests.exceptions.ProxyError("Cannot connect to proxy"),
                                   requests.exceptions.ConnectTimeout("timed out"),
                                   requests.exceptions.ReadTimeout("timed out")])
def test_a_ps1_query_without_network_says_so(monkeypatch, cause):
    def refuse(*a, **kw):
        raise cause
    monkeypatch.setattr(cq.requests, 'get', refuse)
    errors = _query_errors()
    assert len(errors) == 1
    assert errors[0].startswith('Could not reach the PS1 catalog at MAST. Check the internet connection.')
    assert 'Max retries' not in errors[0]


def test_a_ps1_server_error_is_not_called_a_network_problem(monkeypatch):
    def server_error(*a, **kw):
        raise requests.exceptions.HTTPError("503 Server Error")
    monkeypatch.setattr(cq.requests, 'get', server_error)
    assert _query_errors() == ["503 Server Error"]


class _EmptyAnswer:
    """MAST's real answer to a cone with no source: HTTP 200 and no text (checked 2026-10-31)."""
    text = ""

    def raise_for_status(self):
        pass


def test_no_ps1_source_is_an_empty_table_not_an_error(monkeypatch):
    monkeypatch.setattr(cq.requests, 'get', lambda *a, **kw: _EmptyAnswer())
    assert len(cq.ps1cone(13.7, 0.66, 1 / 3600.0)) == 0
    worker = aw.CatalogQueryWorker(13.7, 0.66)
    errors, tables = [], []
    worker.error.connect(errors.append)
    worker.finished.connect(tables.append)
    worker.run()
    assert errors == [] and len(tables) == 1 and len(tables[0]) == 0


def test_a_star_without_a_ps1_source_is_not_a_failed_query(tmp_path, monkeypatch):
    """It used to raise PermissionError on the empty answer, and the export called it a network failure."""
    monkeypatch.setattr(cq.requests, 'get', lambda *a, **kw: _EmptyAnswer())
    out = tmp_path / 'export.csv'
    w = aw.BatchExportWorker([{'kind': 'det', 'x': 1.0, 'y': 2.0, 'ra_deg': 13.7, 'dec_deg': 0.66}], str(out))
    w.run()
    assert w.failed_queries == []
    with open(out, newline='') as fh:
        row = list(csv.DictReader(fh))[0]
    assert row['g_ps1'] == '' and float(row['x']) == 1.0


def test_query_of_a_star_without_a_ps1_source_says_no_match(monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import controllers.astrometry_controller as ac
    from models.astrometry_model import AstrometryModel
    from views.astrometry_view import AstrometryView

    monkeypatch.setattr(cq.requests, 'get', lambda *a, **kw: _EmptyAnswer())
    view = AstrometryView()
    ctrl = ac.AstrometryController(AstrometryModel(), view)
    worker = aw.CatalogQueryWorker(13.7, 0.66)
    worker.finished.connect(ctrl.on_query_finished)
    worker.error.connect(ctrl.on_query_error)
    worker.run()
    assert "No PS1 match found." in view.info_box.toPlainText()


def test_an_error_midway_closes_the_progress_dialog_and_says_so(tmp_path, monkeypatch):
    """Real controller, worker and thread. QProgressDialog closes itself at its maximum, so the
    dialog stayed on screen only when the error came before the last star."""
    import os
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import controllers.astrometry_controller as ac
    from models.astrometry_model import AstrometryModel
    from views.astrometry_view import AstrometryView

    boxes = []
    for kind in ('information', 'warning', 'critical'):
        monkeypatch.setattr(ac.QMessageBox, kind,
                            staticmethod(lambda *a, _k=kind: boxes.append((_k, a[1]))))
    monkeypatch.setattr(aw, 'ps1cone', _ps1_stub)
    monkeypatch.setattr(ac, 'get_export_dir', lambda: str(tmp_path))
    view = AstrometryView()
    # the second star has no pixel x, which raises inside the export loop
    monkeypatch.setattr(view.viewer, 'export_annotated_list', lambda: [
        {'kind': 'det', 'x': 1.0, 'y': 2.0, 'ra_deg': 333.88, 'dec_deg': 51.60},
        {'kind': 'det', 'y': 4.0, 'ra_deg': 333.89, 'dec_deg': 51.61},
        {'kind': 'det', 'x': 5.0, 'y': 6.0, 'ra_deg': 333.90, 'dec_deg': 51.62}])
    ctrl = ac.AstrometryController(AstrometryModel(), view)
    ctrl.on_export_csv()
    t0 = time.time()
    while ctrl.exp_worker.isRunning() and time.time() - t0 < 10:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(20):
        app.processEvents()

    assert ('critical', 'Export Error') in boxes
    assert not ctrl.exp_prog.isVisible()
    assert os.listdir(tmp_path) == []
