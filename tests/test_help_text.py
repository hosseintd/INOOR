"""
The help of the Welcome tab must describe the program as it is: no removed controls, every control
it names exists, and every image loads (a missing file would only show as an empty space).
"""
import glob
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtWidgets

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def help_view():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from views.help_view import HelpView
    yield HelpView()


def _texts(view):
    return [lbl.text() for _, lbl, _, _ in view.text_labels]


def test_no_removed_control_or_old_claim_is_mentioned(help_view):
    text = " ".join(_texts(help_view)).lower()
    for gone in ("gain table", "gaintable", "tracking radius", "catalog overlay", "designation",
                 "previous version", "new version", "no longer"):
        assert gone not in text, gone


def test_the_astrometry_section_says_what_the_tab_does(help_view):
    text = " ".join(_texts(help_view))
    for fact in ("nova.astrometry.net", "green circles are the reference stars of astrometry.net",
                 "red circles are the sources it detected", "Pan-STARRS1 catalog at MAST"):
        assert fact in text, fact


def test_every_control_named_in_quotes_exists(help_view):
    source = "".join(open(p, encoding="utf-8").read()
                     for p in glob.glob(os.path.join(REPO, "views", "**", "*.py"), recursive=True)
                     if not p.endswith("help_view.py"))
    sections = [t for t in _texts(help_view) if t != help_view.lbl_credits.text()]   # the credits are HTML
    named = re.findall(r"(?<!\w)'([^']+)'(?!\w)", " ".join(sections))
    assert len(named) > 15
    for name in named:
        assert name in source, f"the help names '{name}', which no view has"


def test_the_credits_say_what_the_readme_says(help_view):
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    credits = help_view.lbl_credits.text()
    for sentence in ("and Soroush Shakeri (Advisor)",
                     "for their invaluable comments and fruitful discussions.",
                     "We also thank the INO340 Scientific and Technical staff."):
        assert sentence in credits and sentence in readme, sentence
    for name in ("Parisa Hashemi", "Saeid Karimi Mianrudi", "Hamed Altafi", "Reza Rezaei"):
        assert name in credits and name in readme, name
    assert "Team Members" not in credits and "IUT-INO Team" not in credits


def _clipped(view):
    """Paragraphs shorter on screen than their text needs."""
    QtWidgets.QApplication.instance().processEvents()
    return [(lbl.text()[:30], lbl.height(), lbl.heightForWidth(lbl.width()))
            for typ, lbl, _, _ in view.text_labels
            if typ == "p" and lbl.height() < lbl.heightForWidth(lbl.width())]


@pytest.mark.parametrize("size", [(1280, 1392), (1500, 1050), (2560, 1392), (900, 700)])
def test_no_paragraph_loses_its_end(help_view, size):
    """Long paragraphs used to be cut mid-sentence, and the text of the label did not show it."""
    help_view.resize(*size)
    help_view.show()
    assert _clipped(help_view) == []


def test_no_paragraph_loses_its_end_when_zoomed(help_view):
    help_view.resize(1280, 1392)
    help_view.show()
    for step in (help_view._zoom_in, help_view._zoom_in, help_view._zoom_out, help_view._zoom_out,
                 help_view._zoom_out):
        step()
        assert _clipped(help_view) == []
    help_view._zoom_in()


def test_every_image_loads(help_view):
    assert help_view.image_labels
    for _, pixmap, _ in help_view.image_labels:
        assert not pixmap.isNull()
    assert help_view.movie.isValid()
