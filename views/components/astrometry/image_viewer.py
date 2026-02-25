# views/components/astrometry/image_viewer.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QMessageBox, QHBoxLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import os
import math
import warnings

from core.astrometry.image_processor import ImageProcessor

class AstrometryFitsViewer(QWidget):
    """
    GUI widget for displaying a FITS image with WCS-aware annotations.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Top title: filename only
        self.title = QLabel("No image loaded")
        self.title.setStyleSheet("font-weight: bold; color: #d9f0ec;")
        layout.addWidget(self.title, 0, Qt.AlignTop)

        # Matplotlib figure & canvas
        self.fig = Figure(figsize=(6, 6), facecolor='#162a2a')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 1)

        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar { background: #1a3a3a; border: none; padding: 2px; }
            QToolButton { background: #2e6f6f; color: white; border-radius: 3px; margin: 1px; padding: 2px; }
            QToolButton:hover { background: #3d8f8f; }
            QToolButton:pressed { background: #1f4f4f; }
            QLabel { color: #d9f0ec; font-weight: bold; margin-left: 5px; }
        """)
        layout.addWidget(self.toolbar)

        # single axes
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor('#101616')
        
        # Style the axes for visibility in dark theme
        self.ax.tick_params(colors='#e0fcf9', which='both', labelsize=9)
        self.ax.xaxis.label.set_color('#e0fcf9')
        self.ax.yaxis.label.set_color('#e0fcf9')
        for spine in self.ax.spines.values():
            spine.set_color('#2e6f6f')
            spine.set_linewidth(1.5)

        # internals / state
        self.processor = None
        self._original_path = None

        self.catalog_pixcoords_raw = {'ref': [], 'det': []}
        self.catalog_table = None
        self.catalog_indices = {'ref': [], 'det': []}

        self.ref_markers = None
        self.detected_markers = None
        self.info_widget = None

        self.last_picked_pixel = None
        self.last_picked_world = None
        self.last_job_folder = None
        self.last_downsample_factor = 1

        self.rot_k = 0
        self.flip_h = False
        self.flip_v = False

        self.cmap = 'gray'
        self.stretch_mode = 'zscale'
        self.vmin = None
        self.vmax = None

        self.ref_marker_size = 64
        self.ref_marker_linewidth = 2.6
        self.det_marker_size = 48
        self.det_marker_linewidth = 1.8
        self.dedupe_tol_display = 0.6
        self.show_detected = False

        # bottom info
        info_row = QHBoxLayout()
        self.center_label = QLabel("Center: -")
        self.fov_label = QLabel("FoV: -")
        self.center_label.setStyleSheet("color: #d9f0ec; font-size: 10pt;")
        self.fov_label.setStyleSheet("color: #d9f0ec; font-size: 10pt;")
        
        info_row.addWidget(self.center_label)
        info_row.addStretch(1)
        info_row.addWidget(self.fov_label)
        layout.addLayout(info_row)

        self.canvas.mpl_connect('pick_event', self._on_pick_internal)

    def set_info_widget(self, widget):
        self.info_widget = widget

    def set_show_detected(self, show: bool):
        self.show_detected = bool(show)
        try:
            if self.detected_markers is not None:
                self.detected_markers.set_visible(self.show_detected)
                self.canvas.draw_idle()
                return
        except Exception:
            pass
        self.redraw()

    def toggle_detected(self):
        self.set_show_detected(not self.show_detected)

    def load_fits(self, path: str):
        if self.processor:
            try:
                self.processor.close()
            except Exception:
                pass

        self.processor = ImageProcessor(path)
        self._original_path = path

        try:
            self.title.setText(os.path.basename(path))
        except Exception:
            self.title.setText(path)

        self.catalog_pixcoords_raw = {'ref': [], 'det': []}
        self.catalog_table = None
        self.catalog_indices = {'ref': [], 'det': []}
        self.ref_markers = None
        self.detected_markers = None
        self.last_picked_pixel = None
        self.last_picked_world = None

        self.rot_k = 0
        self.flip_h = False
        self.flip_v = False
        self.vmin = None
        self.vmax = None
        self.stretch_mode = 'zscale'

        self.update_field_info()
        self.redraw()

    def update_field_info(self):
        if not self.processor:
            self.center_label.setText("Center: -")
            self.fov_label.setText("FoV: -")
            return

        if getattr(self.processor, 'wcs', None) is None:
            self.center_label.setText("Center: (no WCS)")
            self.fov_label.setText("FoV: (no WCS)")
            return

        ny, nx = self.processor.data.shape
        cx = nx / 2.0
        cy = ny / 2.0
        try:
            center_sky = self.processor.wcs.pixel_to_world(cx, cy)
            ra_deg = float(center_sky.ra.deg)
            dec_deg = float(center_sky.dec.deg)
            ra_hms = center_sky.ra.to_string(unit=u.hourangle, sep=':', precision=2)
            dec_dms = center_sky.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True)
            self.center_label.setText(f"Center: {ra_deg:.6f}° ({ra_hms}), {dec_deg:.6f}° ({dec_dms})")
        except Exception:
            self.center_label.setText("Center: (unavailable)")

        try:
            c1 = self.processor.wcs.pixel_to_world(0, 0)
            c2 = self.processor.wcs.pixel_to_world(nx, ny)
            sep_deg = c1.separation(c2).deg
            self.fov_label.setText(f"FoV: {sep_deg*60.0:.2f} arcmin")
        except Exception:
            self.fov_label.setText("FoV: (unavailable)")

    def redraw(self, vmin=None, vmax=None):
        self.ax.clear()
        
        # Style the axes for visibility in dark theme (must re-apply after clear)
        self.ax.tick_params(colors='#e0fcf9', which='both', labelsize=9)
        self.ax.xaxis.label.set_color('#e0fcf9')
        self.ax.yaxis.label.set_color('#e0fcf9')
        for spine in self.ax.spines.values():
            spine.set_color('#2e6f6f')
            spine.set_linewidth(1.5)

        if self.processor is None:
            self.canvas.draw()
            return

        vmin = vmin if vmin is not None else (self.vmin if self.vmin is not None else None)
        vmax = vmax if vmax is not None else (self.vmax if self.vmax is not None else None)

        img = self.processor.get_display_image(vmin=vmin, vmax=vmax)
        if img is None:
            self.canvas.draw()
            return

        img_disp = np.rot90(img, k=self.rot_k)
        if self.flip_h:
            img_disp = np.fliplr(img_disp)
        if self.flip_v:
            img_disp = np.flipud(img_disp)

        ny_disp, nx_disp = img_disp.shape
        self.ax.imshow(img_disp, origin='lower', cmap=self.cmap, extent=(0, nx_disp, 0, ny_disp), aspect='equal')

        def raw_to_display(x_raw, y_raw):
            ny_raw, nx_raw = self.processor.data.shape
            x = float(x_raw); y = float(y_raw)
            k = self.rot_k % 4
            if k == 0: xd, yd, nx_out, ny_out = x, y, nx_raw, ny_raw
            elif k == 1: xd, yd, nx_out, ny_out = y, (nx_raw - 1.0) - x, ny_raw, nx_raw
            elif k == 2: xd, yd, nx_out, ny_out = (nx_raw - 1.0) - x, (ny_raw - 1.0) - y, nx_raw, ny_raw
            else: xd, yd, nx_out, ny_out = (ny_raw - 1.0) - y, x, ny_raw, nx_raw

            if self.flip_h: xd = (nx_out - 1.0) - xd
            if self.flip_v: yd = (ny_out - 1.0) - yd
            return xd, yd

        def dedupe_raw_list(raw_list, tol_display=self.dedupe_tol_display):
            disp_seen = []; kept_raw = []
            for (xr, yr) in raw_list:
                try: xd, yd = raw_to_display(xr, yr)
                except: continue
                if not (np.isfinite(xd) and np.isfinite(yd)): continue
                duplicate = False
                for (xx, yy) in disp_seen:
                    if abs(xx - xd) <= tol_display and abs(yy - yd) <= tol_display:
                        duplicate = True; break
                if not duplicate:
                    disp_seen.append((xd, yd)); kept_raw.append((xr, yr))
            return disp_seen, kept_raw

        ref_raw = self.catalog_pixcoords_raw.get('ref', []) or []
        det_raw = self.catalog_pixcoords_raw.get('det', []) or []

        ref_disp_pts, ref_kept_raw = dedupe_raw_list(ref_raw)
        det_disp_pts, det_kept_raw = dedupe_raw_list(det_raw)

        self.catalog_pixcoords_raw['ref'] = ref_kept_raw
        self.catalog_pixcoords_raw['det'] = det_kept_raw

        if ref_disp_pts:
            xs = [p[0] for p in ref_disp_pts]; ys = [p[1] for p in ref_disp_pts]
            self.ref_markers = self.ax.scatter(xs, ys, s=self.ref_marker_size, facecolors='none',
                                               edgecolors='lime', linewidths=self.ref_marker_linewidth, picker=6)

        if self.show_detected and det_disp_pts:
            xs = [p[0] for p in det_disp_pts]; ys = [p[1] for p in det_disp_pts]
            self.detected_markers = self.ax.scatter(xs, ys, s=self.det_marker_size, facecolors='none',
                                                    edgecolors='red', linewidths=self.det_marker_linewidth, picker=6)

        self.ax.set_xlim(0, nx_disp); self.ax.set_ylim(0, ny_disp)
        self.canvas.draw_idle()

    def rotate90(self, clockwise: bool = True):
        self.rot_k = (self.rot_k - 1) % 4 if clockwise else (self.rot_k + 1) % 4
        self.redraw()

    def flip_horizontal(self):
        self.flip_h = not self.flip_h; self.redraw()

    def flip_vertical(self):
        self.flip_v = not self.flip_v; self.redraw()

    def set_cmap(self, cmap_name: str):
        self.cmap = str(cmap_name); self.redraw()

    def set_stretch_minmax(self, vmin: float, vmax: float):
        self.vmin = float(vmin); self.vmax = float(vmax)
        self.stretch_mode = 'minmax'; self.redraw(vmin=self.vmin, vmax=self.vmax)

    def set_stretch_zscale(self):
        self.stretch_mode = 'zscale'; self.vmin = None; self.vmax = None; self.redraw()

    def load_astrometry_files(self, rdls_path: str = None, axy_path: str = None):
        if self.processor is None:
            return

        ref_raw = []; det_raw = []
        wcs = getattr(self.processor, 'wcs', None)
        
        if rdls_path and os.path.exists(rdls_path) and wcs is not None:
            try:
                with fits.open(rdls_path, ignore_missing_simple=True) as hd:
                    table_hdu = next((h for h in hd if hasattr(h, 'data') and hasattr(h, 'columns')), None)
                    if table_hdu is not None:
                        cols = [n.lower() for n in table_hdu.columns.names]
                        ra_c = table_hdu.columns.names[cols.index('ra')] if 'ra' in cols else table_hdu.columns.names[0]
                        dec_c = table_hdu.columns.names[cols.index('dec')] if 'dec' in cols else table_hdu.columns.names[1]
                        sky = SkyCoord(table_hdu.data[ra_c], table_hdu.data[dec_c], unit='deg')
                        xpix, ypix = self.processor.wcs.world_to_pixel(sky)
                        ref_raw = list(zip(xpix.tolist(), ypix.tolist()))
            except Exception as e: print("RDLS load fail:", e)

        if axy_path and os.path.exists(axy_path):
            try:
                with fits.open(axy_path, ignore_missing_simple=True) as hd:
                    table_hdu = next((h for h in hd if hasattr(h, 'data') and hasattr(h, 'columns')), None)
                    if table_hdu is not None:
                        xs = table_hdu.data['X']; ys = table_hdu.data['Y']
                        f = getattr(self, 'last_downsample_factor', 1) or 1
                        det_raw = [(float(x) * f, float(y) * f) for x, y in zip(xs.tolist(), ys.tolist())]
            except Exception as e: print("AXY load fail:", e)

        def merge_and_dedupe_raw(existing_list, new_list, tol_raw=0.6):
            out = list(existing_list)
            for x, y in new_list:
                if not any(abs(qx - x) <= tol_raw and abs(qy - y) <= tol_raw for qx, qy in out):
                    out.append((x, y))
            return out

        self.catalog_pixcoords_raw['ref'] = merge_and_dedupe_raw(self.catalog_pixcoords_raw.get('ref', []), ref_raw)
        self.catalog_pixcoords_raw['det'] = merge_and_dedupe_raw(self.catalog_pixcoords_raw.get('det', []), det_raw)
        self.redraw()

    def _on_pick_internal(self, event):
        artist = getattr(event, 'artist', None)
        inds = getattr(event, 'ind', None)
        
        if artist is None or inds is None or len(inds) == 0:
            return
            
        # Handle multiple overlapping points by taking the first one
        idx = int(inds[0])

        if artist == self.detected_markers: kind = 'det'
        elif artist == self.ref_markers: kind = 'ref'
        else: return

        raw_list = self.catalog_pixcoords_raw.get(kind, [])
        if idx >= len(raw_list): return
        x, y = raw_list[idx]
        self.last_picked_pixel = (x, y)
        
        info = f"{'Detected' if kind=='det' else 'Reference'} source\nPixel: ({x:.2f}, {y:.2f})"
        try:
            # Ensure we use raw coordinates
            if self.processor and self.processor.wcs:
                sky = self.processor.wcs.pixel_to_world(x, y)
                self.last_picked_world = sky
                info += f"\nRA: {sky.ra.deg:.6f}\nDec: {sky.dec.deg:.6f}"
            else:
                self.last_picked_world = None
                info += "\n(No WCS available for RA/Dec)"
        except Exception as e:
            print(f"Pick world conversion failed: {e}")
            self.last_picked_world = None
        
        if self.info_widget:
            self.info_widget.setText(info)

    def export_annotated_list(self):
        """
        Return a list of dictionaries with canonical raw x,y.
        """
        rows = []
        for kind in ('ref', 'det'):
            raw_list = self.catalog_pixcoords_raw.get(kind, []) or []
            for (x, y) in raw_list:
                r = {'kind': kind, 'x': float(x), 'y': float(y)}
                try:
                    sky = self.processor.wcs.pixel_to_world(x, y)
                    r['ra_deg'] = float(sky.ra.deg); r['dec_deg'] = float(sky.dec.deg)
                except:
                    r['ra_deg'] = None; r['dec_deg'] = None
                rows.append(r)
        return rows

    def close(self):
        if self.processor: self.processor.close()
