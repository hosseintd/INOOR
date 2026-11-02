"""
All three radius diagnostics for one star in one frame, side by side.

The three plots answer three different questions and INOOR picks the aperture
from one of them depending on the active policy, so being able to see them
together is what lets a user check that the chosen radius is sensible:

  radial profile  -> how wide is the star (the PSF itself)
  SNR vs radius   -> which radius maximises signal-to-noise for THIS star
  curve of growth -> which radius contains a given fraction of its light

The right panel is the one the active policy reads its radius from; it is
labelled so that is not left implicit.
"""
import csv
import os
from PyQt5 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from utils.gui_helpers import plot_profile_and_snr, plot_growth_curve
from views.theme import set_variant, save_publication_figure, style_figure


class RadialPlotsDialog(QtWidgets.QDialog):
    def __init__(self, parent, data, title_suffix="", policy_label=""):
        """
        data: dict with rp_radii, rp_profile, radii, snrs, r_best, r_used,
              cog_radii, cog_frac, growth_target; optional frame, star_xy
              (only written into the CSV headers)
        """
        super().__init__(parent)
        self.data = data or {}
        self.setWindowTitle(f"Radial diagnostics {title_suffix}".strip())
        self.resize(1180, 480)
        # Styling comes from the application stylesheet (views/theme.py).
        self._build(policy_label)
        self._draw()

    def _build(self, policy_label):
        root = QtWidgets.QVBoxLayout(self)

        hdr = QtWidgets.QLabel(
            f"Aperture radius in use: <b>{self.data.get('r_used', float('nan')):.2f} px</b>"
            + (f" &nbsp;|&nbsp; chosen by: <b>{policy_label}</b>" if policy_label else ""))
        hdr.setTextFormat(QtCore.Qt.RichText)
        root.addWidget(hdr)

        row = QtWidgets.QHBoxLayout()
        self.figs = {}
        for key, caption in (("profile", "Radial profile (PSF shape)"),
                             ("snr", "SNR vs radius"),
                             ("growth", "Curve of growth")):
            col = QtWidgets.QVBoxLayout()
            lbl = QtWidgets.QLabel(caption)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            set_variant(lbl, "heading")
            fig = Figure(figsize=(3.8, 3.0), tight_layout=True)
            style_figure(fig)
            canvas = FigureCanvas(fig)
            col.addWidget(lbl)
            col.addWidget(canvas)
            self.figs[key] = (fig, canvas)
            row.addLayout(col)
        root.addLayout(row)

        btns = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save all three...")
        self.btn_csv = QtWidgets.QPushButton("Export data (CSV)...")
        self.btn_csv.setToolTip("Write the numbers behind the three plots, one CSV per plot,\n"
                                "so they can be re-plotted elsewhere.")
        self.btn_close = QtWidgets.QPushButton("Close")
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_csv)
        btns.addStretch()
        btns.addWidget(self.btn_close)
        root.addLayout(btns)

        self.btn_save.clicked.connect(self.on_save)
        self.btn_csv.clicked.connect(self.on_export_csv)
        self.btn_close.clicked.connect(self.accept)

    def _draw(self):
        d = self.data
        plot_profile_and_snr(self.figs["profile"][0], self.figs["snr"][0],
                             d.get("rp_radii"), d.get("rp_profile"),
                             d.get("radii"), d.get("snrs"), d.get("r_best"),
                             r_used=d.get("r_used"), frame_fwhm=d.get("frame_fwhm"))
        plot_growth_curve(self.figs["growth"][0], d.get("cog_radii"), d.get("cog_frac"),
                          r_used=d.get("r_used"), target=d.get("growth_target"))
        for fig, canvas in self.figs.values():
            canvas.draw()

    def on_save(self):
        base, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save radial diagnostics (a suffix is added per plot)",
            "radial_diagnostics.png", "PNG (*.png);;PDF (*.pdf)")
        if not base:
            return
        root, ext = os.path.splitext(base)
        ext = ext or ".png"
        written = []
        for key, (fig, _c) in self.figs.items():
            path = f"{root}_{key}{ext}"
            try:
                # Black on white regardless of the screen theme.
                save_publication_figure(fig, path, dpi=300)
                written.append(os.path.basename(path))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Could not save",
                                              f"{path}\n\n{e}")
                return
        QtWidgets.QMessageBox.information(self, "Saved", "Wrote:\n" + "\n".join(written))

    def on_export_csv(self):
        # One file per plot, named like the PNG export: the three curves are
        # sampled on different radius grids, so they do not share one table.
        d = self.data
        base, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export radial diagnostics data (a suffix is added per plot)",
            "radial_diagnostics.csv", "CSV (*.csv)")
        if not base:
            return
        root = os.path.splitext(base)[0]

        def num(v):
            return "n/a" if v is None else f"{float(v):.4f}"

        common = []
        if d.get("frame"):
            common.append(f"frame: {d['frame']}")
        if d.get("star_xy") is not None:
            x, y = d["star_xy"]
            common.append(f"star x, y (px): {float(x):.3f}, {float(y):.3f}")
        common.append(f"aperture radius in use (px): {num(d.get('r_used'))}")
        common.append(f"SNR peak radius (px): {num(d.get('r_best'))}")

        tables = (
            ("profile", d.get("rp_radii"), d.get("rp_profile"),
             ("radius_px", "mean_pixel_value"),
             ["mean pixel value in 1-px annuli, image units, NOT background-subtracted"]),
            ("snr", d.get("radii"), d.get("snrs"),
             ("radius_px", "snr"), []),
            ("growth", d.get("cog_radii"), d.get("cog_frac"),
             ("radius_px", "enclosed_fraction"),
             ["enclosed_fraction: background-subtracted, 1.0 at the outermost radius",
              f"growth target fraction: {num(d.get('growth_target'))}"]),
        )
        written, skipped = [], []
        for key, xs, ys, cols, notes in tables:
            if xs is None or ys is None or len(xs) == 0:
                skipped.append(key)
                continue
            path = f"{root}_{key}.csv"
            try:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    for line in common + notes:
                        fh.write(f"# {line}\n")
                    w = csv.writer(fh)
                    w.writerow(cols)
                    for xv, yv in zip(xs, ys):
                        w.writerow([f"{float(xv):.4f}", f"{float(yv):.6g}"])
                written.append(os.path.basename(path))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Could not save", f"{path}\n\n{e}")
                return
        msg = "Wrote:\n" + "\n".join(written)
        if skipped:
            msg += "\n\nNo data, not written: " + ", ".join(skipped)
        QtWidgets.QMessageBox.information(self, "Exported", msg)
