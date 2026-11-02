"""Single source of truth for every colour and font in the GUI.

Why this file exists
--------------------
Before this, each view called ``setStyleSheet`` on itself with its own hard-coded
hex values (~310 literals across 23 files). A widget-level stylesheet always wins
over the application-level one, so no global theme could ever take effect --
switching themes was impossible by construction. Views now style nothing; they
only mark widgets with a ``variant`` property (see below) and the application
stylesheet built here does the rest.

Adding a theme means adding one entry to ``THEMES``. Nothing else changes.

Semantic tokens, not colour names
---------------------------------
Tokens are named for their ROLE (``bg_input``, ``danger``) rather than their hue,
so the night theme can be entirely red without any token name becoming a lie.

Button variants
---------------
Views set a dynamic property instead of a stylesheet::

    btn.setProperty("variant", "primary")   # the one main action of a panel
    btn.setProperty("variant", "danger")    # destructive: clear/remove/delete

Anything unmarked gets the neutral treatment. If you change a variant after the
widget is shown, call ``repolish(widget)`` -- Qt does not re-evaluate property
selectors on its own.

Matplotlib
----------
Figures are not Qt widgets and ignore the stylesheet, so they read their colours
from ``mpl_colors()`` and re-read them on ``theme_manager.changed``.
"""

import sys

from PyQt5 import QtCore, QtGui, QtWidgets

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

#: Every theme must define exactly these keys. ``build_qss`` formats the
#: stylesheet template against this dict, so a missing key fails loudly at
#: startup rather than silently dropping a rule.
TOKEN_NAMES = (
    "bg_base", "bg_surface", "bg_raised", "bg_input", "bg_canvas",
    "border", "border_muted",
    "text", "text_muted", "text_heading", "text_on_accent",
    "accent", "accent_hover", "accent_press", "accent_disabled",
    "danger", "danger_hover", "warning", "success",
    "focus", "selection", "selection_text",
    "disabled_bg", "disabled_text",
)


DARK = {
    "label": "Observatory Dark",
    "is_dark": True,
    # Deepest layer is the window itself; panels sit one step lighter so the
    # three-column layouts read as cards instead of one flat field.
    "bg_base": "#101f1e",
    "bg_surface": "#162a2a",
    "bg_raised": "#1c3735",
    "bg_input": "#0d1a19",
    "bg_canvas": "#101616",
    # 3.00:1 against bg_surface, the WCAG 1.4.11 floor for a control boundary.
    "border": "#327a7a",
    "border_muted": "#24514f",
    "text": "#d9f0ec",
    "text_muted": "#93b8b3",
    "text_heading": "#7fe8dd",
    "text_on_accent": "#ffffff",
    "accent": "#0f7a73",
    # Deliberately only a little lighter than accent: anything brighter drops
    # white button text below 4.5:1.
    "accent_hover": "#11847e",
    "accent_press": "#0a524d",
    "accent_disabled": "#2a3f3e",
    "danger": "#ff5c52",
    "danger_hover": "#c2453a",
    "warning": "#ffcc66",
    "success": "#43a06b",
    "focus": "#3df2e5",
    "selection": "#17605c",
    "selection_text": "#ffffff",
    "disabled_bg": "#22302f",
    "disabled_text": "#6b807e",
}


# Deep red on black. Rod cells -- the ones carrying dark-adapted vision -- peak
# near 498 nm and are nearly blind above ~620 nm, so a red-only display leaves
# dark adaptation intact where any white or cyan light would bleach it and cost
# 20-30 minutes of re-adaptation. Every token here is kept free of green and
# blue for that reason, and overall luminance is held low: hue alone does not
# save dark adaptation if the screen is bright.
#
# Trade-off worth knowing: ``text`` sits at ~4.7:1 on black so the theme stays
# legible in a lit room too. A dark-adapted eye would be comfortable with
# something dimmer, but that would be unusable anywhere else.
NIGHT_RED = {
    "label": "Night (red)",
    "is_dark": True,
    "bg_base": "#000000",
    "bg_surface": "#0c0201",
    "bg_raised": "#1a0503",
    "bg_input": "#050101",
    "bg_canvas": "#000000",
    # Knowingly below the 3:1 WCAG floor for control boundaries (~2.1:1).
    # Raising it to 3:1 needs roughly #ab2b1e, which lifts the luminance of
    # every border on screen and works directly against the one thing this
    # theme is for. Dark-adapted vision is orders of magnitude more sensitive
    # than the daylight viewing WCAG models, so the trade is made on purpose
    # and only here -- the dark and light themes both meet the floor.
    "border": "#7d1f15",
    "border_muted": "#4a0f0a",
    "text": "#e04030",
    "text_muted": "#d54133",
    "text_heading": "#ff5a42",
    "text_on_accent": "#ffd0c6",
    "accent": "#6e1109",
    "accent_hover": "#94180d",
    "accent_press": "#4a0b06",
    "accent_disabled": "#2a0806",
    # Hue is unavailable as a signal here, so severity is carried by brightness.
    "danger": "#ff3a1e",
    "danger_hover": "#ff5c42",
    "warning": "#ff7a52",
    "success": "#d73f26",
    "focus": "#ff6a4d",
    "selection": "#4a0b06",
    "selection_text": "#ffd0c6",
    "disabled_bg": "#170403",
    "disabled_text": "#8e2a20",
}


LIGHT = {
    "label": "Daylight",
    "is_dark": False,
    "bg_base": "#eaf0ef",
    "bg_surface": "#ffffff",
    "bg_raised": "#dfe9e7",
    "bg_input": "#ffffff",
    "bg_canvas": "#f6faf9",
    "border": "#749c97",
    "border_muted": "#cfdedc",
    "text": "#152a2c",
    "text_muted": "#55706d",
    "text_heading": "#0a5d57",
    "text_on_accent": "#ffffff",
    "accent": "#0f7a73",
    "accent_hover": "#12847c",
    "accent_press": "#0a524d",
    "accent_disabled": "#c3d3d1",
    "danger": "#a72820",
    "danger_hover": "#c2352b",
    "warning": "#8a5a00",
    "success": "#1c6b46",
    "focus": "#0f7a73",
    "selection": "#bde2de",
    "selection_text": "#0d2b29",
    "disabled_bg": "#e3eae9",
    "disabled_text": "#7c918f",
}


THEMES = {
    "dark": DARK,
    "night": NIGHT_RED,
    "light": LIGHT,
}

DEFAULT_THEME = "dark"


#: Preferred UI faces per platform, best first. The old code hard-coded Arial,
#: which is nobody's system font on Windows 10/11.
_FONT_CANDIDATES = {
    "win32": ("Segoe UI", "Tahoma", "Arial"),
    "darwin": ("SF Pro Text", "Helvetica Neue", "Helvetica"),
}


def ui_font():
    """A QFont for the whole application.

    This is applied with ``QApplication.setFont`` rather than a ``font-size``
    rule in the stylesheet. A stylesheet font-size beats any later
    ``widget.setFont()``, which would have silently broken the Help tab's
    zoom buttons; an application font does not.
    """
    available = set(QtGui.QFontDatabase().families())
    for name in _FONT_CANDIDATES.get(sys.platform, ("Cantarell", "DejaVu Sans")):
        if name in available:
            return QtGui.QFont(name, 10)
    return QtGui.QFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont))


# --------------------------------------------------------------------------
# Stylesheet
# --------------------------------------------------------------------------

# Sizes are in pt, not px, so Qt scales them with the user's display scaling
# setting instead of rendering tiny on a 4K panel.
_QSS = """
QWidget {{
    background: {bg_base};
    color: {text};
}}

/* Panels and canvases carry their own surface colour; plain QFrame containers
   used purely for layout must stay transparent or they punch holes in it. */
QFrame, QScrollArea, QSplitter, QStackedWidget {{
    background: transparent;
}}

QMainWindow, QDialog {{
    background: {bg_base};
}}

QToolTip {{
    background: {bg_raised};
    color: {text};
    border: 1px solid {border};
    padding: 4px 6px;
}}

/* ---------------- tabs ---------------- */
/* Previously unstyled, so the tab strip rendered in the native light Windows
   style across the top of an otherwise dark window. */
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 4px;
    background: {bg_surface};
    top: -1px;
}}

QTabBar {{
    background: {bg_base};
}}

QTabBar::tab {{
    background: {bg_base};
    color: {text_muted};
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
    padding: 7px 18px;
    margin-right: 2px;
}}

QTabBar::tab:hover:!selected {{
    background: {bg_raised};
    color: {text};
}}

/* Deliberately not bold: Qt sizes each tab from the unselected font, so making
   the selected one bold widens the text past the tab and clips it
   ("Multi Photometry" rendered as "Multi Photometr"). The surface change,
   heading colour and accent underline mark selection well enough. */
QTabBar::tab:selected {{
    background: {bg_surface};
    color: {text_heading};
    border: 1px solid {border};
    border-bottom: 2px solid {accent_hover};
}}

/* ---------------- grouping ---------------- */
QGroupBox {{
    background: {bg_surface};
    border: 1px solid {border_muted};
    border-radius: 5px;
    /* Room for the title to sit above the border without colliding with it.
       Padding is kept tight: these panels are narrow and every horizontal
       pixel spent here pushes the controls into a scrollbar. */
    margin-top: 13px;
    padding: 6px 5px 5px 5px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: {text_heading};
    background: transparent;
}}

QGroupBox:disabled, QGroupBox:disabled::title {{
    color: {disabled_text};
}}

/* ---------------- buttons ---------------- */
QPushButton {{
    background: {bg_raised};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 5px 8px;
    min-height: 18px;
}}

QPushButton:hover {{
    background: {accent};
    border-color: {accent_hover};
    color: {text_on_accent};
}}

QPushButton:pressed {{
    background: {accent_press};
    border-color: {accent_press};
}}

QPushButton:focus {{
    border: 1px solid {focus};
}}

QPushButton:disabled {{
    background: {disabled_bg};
    color: {disabled_text};
    border-color: {border_muted};
}}

/* The one action a panel exists to perform. */
QPushButton[variant="primary"] {{
    background: {accent};
    color: {text_on_accent};
    border: 1px solid {accent_hover};
    font-weight: bold;
    padding: 7px 10px;
}}

QPushButton[variant="primary"]:hover {{
    background: {accent_hover};
}}

QPushButton[variant="primary"]:pressed {{
    background: {accent_press};
}}

QPushButton[variant="primary"]:disabled {{
    background: {accent_disabled};
    color: {disabled_text};
    border-color: {border_muted};
}}

/* Destructive. These used to be visually identical to "Add", which is how a
   tired operator clears a 200-frame list at 3 a.m. */
QPushButton[variant="danger"] {{
    background: transparent;
    color: {danger};
    border: 1px solid {danger};
}}

QPushButton[variant="danger"]:hover {{
    background: {danger};
    color: {text_on_accent};
}}

QPushButton[variant="danger"]:pressed {{
    background: {danger_hover};
    color: {text_on_accent};
}}

QPushButton[variant="success"] {{
    background: {success};
    color: {text_on_accent};
    border: 1px solid {success};
    font-weight: bold;
}}

QToolButton {{
    background: {bg_raised};
    color: {text};
    border: 1px solid {border_muted};
    border-radius: 3px;
    padding: 4px 6px;
}}

QToolButton:hover {{
    background: {accent};
    color: {text_on_accent};
}}

QToolButton:pressed {{
    background: {accent_press};
}}

/* The collapsible-section header is a QToolButton but should read as a header,
   not a button. */
QToolButton[variant="section"] {{
    background: {bg_raised};
    color: {text_heading};
    border: none;
    border-radius: 3px;
    font-weight: bold;
    text-align: left;
    padding: 6px 8px;
}}

QToolButton[variant="section"]:hover {{
    background: {selection};
    color: {text};
}}

/* ---------------- inputs ---------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit,
QDateTimeEdit {{
    background: {bg_input};
    color: {text};
    border: 1px solid {border_muted};
    border-radius: 3px;
    padding: 4px 6px;
    selection-background-color: {selection};
    selection-color: {selection_text};
}}

QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {border};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {focus};
}}

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled, QTextEdit:disabled {{
    background: {disabled_bg};
    color: {disabled_text};
    border-color: {border_muted};
}}

QComboBox QAbstractItemView {{
    background: {bg_input};
    color: {text};
    border: 1px solid {border};
    selection-background-color: {selection};
    selection-color: {selection_text};
    outline: none;
}}

/* ---------------- checkboxes and radios ---------------- */
/* These were left unstyled outside the Astrometry tab, so Qt drew the native
   white-and-blue Windows indicator on a dark teal panel. */
QCheckBox, QRadioButton {{
    background: transparent;
    color: {text};
    spacing: 7px;
    padding: 2px 0;
}}

QCheckBox:disabled, QRadioButton:disabled {{
    color: {disabled_text};
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    background: {bg_input};
    border: 1px solid {border};
}}

QCheckBox::indicator {{
    border-radius: 3px;
}}

QRadioButton::indicator {{
    border-radius: 8px;
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {focus};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {accent_hover};
    border: 1px solid {accent_hover};
}}

QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background: {disabled_bg};
    border-color: {border_muted};
}}

/* Checkbox whose label is a caution, e.g. the cosmic-ray filter that also
   clips real stellar cores. */
QCheckBox[variant="warning"] {{
    color: {warning};
}}

/* ---------------- lists, trees, tables ---------------- */
QListWidget, QListView, QTreeWidget, QTreeView, QTableWidget, QTableView {{
    background: {bg_input};
    color: {text};
    border: 1px solid {border_muted};
    border-radius: 3px;
    alternate-background-color: {bg_surface};
    outline: none;
}}

QListWidget::item, QTreeWidget::item, QTreeView::item {{
    padding: 3px 4px;
    border: none;
}}

QListWidget::item:hover, QTreeWidget::item:hover, QTreeView::item:hover {{
    background: {bg_raised};
}}

QListWidget::item:selected, QListView::item:selected,
QTreeWidget::item:selected, QTreeView::item:selected,
QTableWidget::item:selected, QTableView::item:selected {{
    background: {selection};
    color: {selection_text};
}}

QHeaderView {{
    background: {bg_raised};
}}

QHeaderView::section {{
    background: {bg_raised};
    color: {text_heading};
    border: none;
    border-right: 1px solid {border_muted};
    border-bottom: 1px solid {border};
    padding: 5px 6px;
    font-weight: bold;
}}

QTableCornerButton::section {{
    background: {bg_raised};
    border: none;
}}

/* ---------------- scrollbars ---------------- */
QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 0;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 11px;
    margin: 0;
}}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {border_muted};
    border-radius: 5px;
    min-height: 28px;
    min-width: 28px;
}}

QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {border};
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}

QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* ---------------- splitters ---------------- */
QSplitter::handle {{
    background: {border_muted};
}}

QSplitter::handle:hover {{
    background: {accent_hover};
}}

QSplitter::handle:horizontal {{
    width: 4px;
}}

QSplitter::handle:vertical {{
    height: 4px;
}}

/* ---------------- labels ---------------- */
QLabel {{
    background: transparent;
    color: {text};
}}

QLabel[variant="title"] {{
    color: {text_heading};
    font-size: 13pt;
    font-weight: bold;
}}

/* Colour only, no font-size: for labels whose size is set in code, such as
   the Help tab's zoomable banner title. */
QLabel[variant="brand"] {{
    color: {text_heading};
}}

QLabel[variant="heading"] {{
    color: {text_heading};
    font-size: 11pt;
    font-weight: bold;
}}

QLabel[variant="muted"] {{
    color: {text_muted};
}}

QLabel[variant="filename"] {{
    color: {text_heading};
    font-size: 11pt;
    font-weight: bold;
}}

QLabel[variant="value"] {{
    color: {text};
    font-weight: bold;
}}

QLabel[variant="warning"] {{
    color: {warning};
}}

QLabel[variant="danger"] {{
    color: {danger};
}}

QLabel[variant="success"] {{
    color: {success};
}}

/* ---------------- progress, menus, misc ---------------- */
QProgressBar {{
    background: {bg_input};
    color: {text};
    border: 1px solid {border_muted};
    border-radius: 3px;
    text-align: center;
    min-height: 16px;
}}

QProgressBar::chunk {{
    background: {accent};
    border-radius: 2px;
}}

QMenuBar {{
    background: {bg_base};
    color: {text};
}}

QMenuBar::item:selected {{
    background: {selection};
    color: {selection_text};
}}

QMenu {{
    background: {bg_surface};
    color: {text};
    border: 1px solid {border};
}}

QMenu::item:selected {{
    background: {selection};
    color: {selection_text};
}}

QStatusBar {{
    background: {bg_base};
    color: {text_muted};
    border-top: 1px solid {border_muted};
}}

QToolBar {{
    background: {bg_raised};
    border: none;
    padding: 2px;
    spacing: 2px;
}}

QSlider::groove:horizontal {{
    background: {bg_input};
    height: 5px;
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {accent_hover};
    border: 1px solid {border};
    width: 13px;
    margin: -5px 0;
    border-radius: 7px;
}}
"""


def build_qss(theme_key=DEFAULT_THEME):
    """Return the complete application stylesheet for one theme."""
    tokens = THEMES.get(theme_key, THEMES[DEFAULT_THEME])
    missing = [n for n in TOKEN_NAMES if n not in tokens]
    if missing:
        raise KeyError(f"theme {theme_key!r} is missing tokens: {missing}")
    return _QSS.format(**{n: tokens[n] for n in TOKEN_NAMES})


# --------------------------------------------------------------------------
# Matplotlib side
# --------------------------------------------------------------------------

def _dimmed_gray():
    """Grey ramp that tops out below white, for the night theme.

    The image stays greyscale -- judging frame quality needs real luminance
    ordering, and a red-mapped star field is much harder to assess -- but a
    full-white core on an otherwise near-black screen would undo the dark
    adaptation the theme exists to protect, so the top end is clipped.
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import cm

    base = cm.get_cmap("gray")
    return LinearSegmentedColormap.from_list(
        "gray_dim", base(np.linspace(0.0, 0.62, 256))
    )


_MPL_CACHE = {}


def mpl_colors(theme_key=None):
    """Colours for matplotlib figures, which ignore Qt stylesheets entirely.

    ``image_cmap`` is the default colormap for FITS display; ``series`` are the
    plot line colours, ordered so the first three stay distinguishable for the
    most common colour-vision deficiencies.
    """
    key = theme_key or theme_manager.current()
    if key in _MPL_CACHE:
        return _MPL_CACHE[key]

    t = THEMES.get(key, THEMES[DEFAULT_THEME])
    out = {
        "fig_bg": t["bg_surface"],
        "ax_bg": t["bg_canvas"],
        "text": t["text"],
        "muted": t["text_muted"],
        "spine": t["border"],
        "grid": t["border_muted"],
        "accent": t["accent_hover"],
        "danger": t["danger"],
        "warning": t["warning"],
        "success": t["success"],
    }

    if key == "night":
        out["image_cmap"] = _dimmed_gray()
        # No hue budget in red mode: separate series by lightness instead.
        out["series"] = ["#ff6a4d", "#c2331f", "#ff9d85", "#8f1f12", "#ffd0c6"]
        # Aperture and sky-annulus circles drawn on the FITS image: light reds,
        # told apart by lightness and by the annulus' dashes.
        out["aperture"], out["annulus"] = "#ff9d85", "#ff6a4d"
    else:
        out["image_cmap"] = "gray"
        # Colour-blind-safe triad already used for this project's figures.
        out["series"] = ["#8E44AD", "#16A085", "#E67E22", "#3498DB", "#C0392B"]
        # Saturated hues that stay visible on any grey level of the image.
        out["aperture"], out["annulus"] = "cyan", "yellow"

    _MPL_CACHE[key] = out
    return out


def style_axes(ax, grid=True):
    """Apply the current theme to one matplotlib Axes."""
    c = mpl_colors()
    ax.set_facecolor(c["ax_bg"])
    ax.tick_params(axis="both", colors=c["text"], labelsize=8, which="both")
    for spine in ax.spines.values():
        spine.set_color(c["spine"])
    ax.xaxis.label.set_color(c["text"])
    ax.yaxis.label.set_color(c["text"])
    ax.title.set_color(c["text"])
    if grid:
        ax.grid(True, ls=":", color=c["grid"], alpha=0.55)


def style_figure(fig):
    """Apply the current theme to a matplotlib Figure's background."""
    fig.patch.set_facecolor(mpl_colors()["fig_bg"])


#: Ink and paper for exported figures. Fixed, not themed -- see below.
PUBLICATION_FG = "#000000"
PUBLICATION_BG = "#ffffff"


def paper_figure_colors():
    """Colours for the two publication-style plot dialogs on screen.

    The light-curve dialog carries an explicit "keep the white background of
    plot" note from the operator, and the extinction fit matches it, so these
    two figures stay white on screen under the dark and light themes exactly as
    before.

    The night theme is the single exception. A full white panel is precisely
    the light that mode exists to keep out of a dark-adapted eye, so there the
    figure follows the theme. What gets written to disk is unaffected either
    way -- see save_publication_figure.
    """
    if theme_manager.current() == "night":
        c = mpl_colors()
        return {"fig_bg": c["fig_bg"], "ax_bg": c["ax_bg"], "ink": c["text"],
                "grid": c["grid"], "spine": c["spine"]}
    return {"fig_bg": PUBLICATION_BG, "ax_bg": PUBLICATION_BG,
            "ink": PUBLICATION_FG, "grid": "#b0b0b0", "spine": PUBLICATION_FG}


def save_publication_figure(fig, path, dpi=300, **kwargs):
    """Save `fig` on white paper with black ink, whatever the screen theme is.

    These plots end up in a paper, so the exported file must not change when
    the operator switches themes -- but the on-screen copy should follow the
    theme, because a full-white panel at the telescope is exactly what the
    night mode exists to prevent. So: theme on screen, print colours in the
    file. Axis text is recoloured for the write and restored afterwards.
    """
    from matplotlib.colors import to_rgba
    muted = to_rgba(mpl_colors()["muted"])
    patches = []
    inked = []
    for ax in fig.get_axes():
        patches.append((ax, ax.get_facecolor(),
                        [s.get_edgecolor() for s in ax.spines.values()]))
        # Annotations drawn in the theme's muted colour (the SNR peak and its k
        # label) are made to recede on a dark screen and vanish on white paper.
        for artist in list(ax.texts) + list(ax.lines):
            if to_rgba(artist.get_color()) == muted:
                inked.append((artist, artist.get_color()))
                artist.set_color(PUBLICATION_FG)
        ax.set_facecolor(PUBLICATION_BG)
        for s in ax.spines.values():
            s.set_color(PUBLICATION_FG)
        ax.tick_params(colors=PUBLICATION_FG, which="both")
        for text in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                     + ax.get_xticklabels() + ax.get_yticklabels()):
            text.set_color(PUBLICATION_FG)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(PUBLICATION_BG)
            leg.get_frame().set_edgecolor(PUBLICATION_FG)
            for txt in leg.get_texts():
                txt.set_color(PUBLICATION_FG)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=PUBLICATION_BG, edgecolor=PUBLICATION_BG, **kwargs)
    finally:
        c = mpl_colors()
        for artist, col in inked:
            artist.set_color(col)
        for ax, face, spine_colors in patches:
            ax.set_facecolor(face)
            for s, sc in zip(ax.spines.values(), spine_colors):
                s.set_color(sc)
            ax.tick_params(colors=c["text"], which="both")
            for text in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                         + ax.get_xticklabels() + ax.get_yticklabels()):
                text.set_color(c["text"])
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(c["fig_bg"])
                leg.get_frame().set_edgecolor(c["spine"])
                for txt in leg.get_texts():
                    txt.set_color(c["text"])
        fig.canvas.draw_idle()


# --------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------

class _ThemeManager(QtCore.QObject):
    """Applies a theme app-wide and tells non-Qt-styled widgets to catch up.

    Anything that cannot be reached by a stylesheet -- matplotlib figures, the
    navigation toolbar's baked-in dark icons -- connects to ``changed``.
    """

    changed = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._current = DEFAULT_THEME

    def current(self):
        return self._current

    def tokens(self):
        return THEMES[self._current]

    def color(self, token):
        """One token's colour, for the rare place that needs a raw value."""
        return THEMES[self._current].get(token, "#000000")

    def is_dark(self):
        return THEMES[self._current]["is_dark"]

    def load_saved(self):
        """The theme chosen last session, or the default on first run."""
        s = QtCore.QSettings("INO", "INOOR")
        key = s.value("ui/theme", DEFAULT_THEME)
        return key if key in THEMES else DEFAULT_THEME

    def apply(self, theme_key, save=True):
        if theme_key not in THEMES:
            theme_key = DEFAULT_THEME
        self._current = theme_key

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setFont(ui_font())
            # The palette matters for widgets Qt draws without consulting the
            # stylesheet -- notably QMessageBox icons and native file dialogs.
            app.setPalette(build_palette(theme_key))
            app.setStyleSheet(build_qss(theme_key))

        if save:
            QtCore.QSettings("INO", "INOOR").setValue("ui/theme", theme_key)

        self.changed.emit(theme_key)


def build_palette(theme_key=DEFAULT_THEME):
    t = THEMES.get(theme_key, THEMES[DEFAULT_THEME])
    p = QtGui.QPalette()
    c = QtGui.QColor

    p.setColor(QtGui.QPalette.Window, c(t["bg_base"]))
    p.setColor(QtGui.QPalette.WindowText, c(t["text"]))
    p.setColor(QtGui.QPalette.Base, c(t["bg_input"]))
    p.setColor(QtGui.QPalette.AlternateBase, c(t["bg_surface"]))
    p.setColor(QtGui.QPalette.Text, c(t["text"]))
    p.setColor(QtGui.QPalette.Button, c(t["bg_raised"]))
    p.setColor(QtGui.QPalette.ButtonText, c(t["text"]))
    p.setColor(QtGui.QPalette.BrightText, c(t["danger"]))
    p.setColor(QtGui.QPalette.Highlight, c(t["selection"]))
    p.setColor(QtGui.QPalette.HighlightedText, c(t["selection_text"]))
    p.setColor(QtGui.QPalette.ToolTipBase, c(t["bg_raised"]))
    p.setColor(QtGui.QPalette.ToolTipText, c(t["text"]))
    p.setColor(QtGui.QPalette.Link, c(t["accent_hover"]))
    p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, c(t["disabled_text"]))
    p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, c(t["disabled_text"]))
    p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, c(t["disabled_text"]))
    return p


#: Module-level singleton. Views import this, not a copy.
theme_manager = _ThemeManager()


# --------------------------------------------------------------------------
# Helpers for views
# --------------------------------------------------------------------------

def set_variant(widget, variant):
    """Tag a widget for the stylesheet, e.g. ``set_variant(btn, "danger")``."""
    widget.setProperty("variant", variant)
    return widget


def repolish(widget):
    """Force Qt to re-evaluate property selectors after a variant changes."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _tinted(pixmap, color):
    """Copy of `pixmap` with its RGB replaced by `color`, alpha preserved.

    Works because these toolbar glyphs are flat and monochrome.
    """
    out = QtGui.QPixmap(pixmap.size())
    out.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(out)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
    painter.fillRect(out.rect(), QtGui.QColor(color))
    painter.end()
    return out


def recolor_toolbar_icons(toolbar):
    """Tint a matplotlib NavigationToolbar's icons to the current theme.

    The icons ship as dark pixmaps meant for a light background; on the dark
    themes they were all but invisible.

    The disabled variant is supplied explicitly rather than left to Qt, which
    derives it by desaturating to grey -- fine normally, but in the night theme
    grey is exactly the non-red light the theme exists to avoid.
    """
    normal = theme_manager.color("text")
    disabled = theme_manager.color("disabled_text")
    for action in toolbar.actions():
        icon = action.icon()
        if icon.isNull():
            continue
        sizes = icon.availableSizes()
        pm = icon.pixmap(sizes[0] if sizes else QtCore.QSize(24, 24))
        if pm.isNull():
            continue
        out = QtGui.QIcon()
        out.addPixmap(_tinted(pm, normal), QtGui.QIcon.Normal)
        out.addPixmap(_tinted(pm, disabled), QtGui.QIcon.Disabled)
        action.setIcon(out)


class ThemeSwitcher(QtWidgets.QWidget):
    """Compact theme picker, meant for the tab bar's corner."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 8, 2)
        lay.setSpacing(6)

        label = QtWidgets.QLabel("Theme:")
        set_variant(label, "muted")
        lay.addWidget(label)

        self.combo = QtWidgets.QComboBox()
        for key, t in THEMES.items():
            self.combo.addItem(t["label"], key)
        self.combo.setToolTip(
            "Observatory Dark - general use.\n"
            "Night (red) - preserves dark adaptation at the telescope; the FITS\n"
            "  image stays greyscale but dimmed.\n"
            "Daylight - daytime use, projectors, and screenshots for papers."
        )
        self.combo.setMinimumWidth(150)
        lay.addWidget(self.combo)

        self._sync()
        self.combo.currentIndexChanged.connect(self._on_changed)
        # The theme can also change from elsewhere -- restored from settings at
        # startup, or a second switcher -- so follow it rather than assume this
        # combo is the only source of truth.
        theme_manager.changed.connect(self._sync)

    def _sync(self, *_):
        i = self.combo.findData(theme_manager.current())
        if i >= 0 and i != self.combo.currentIndex():
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(i)
            self.combo.blockSignals(False)

    def _on_changed(self, _idx):
        theme_manager.apply(self.combo.currentData())
