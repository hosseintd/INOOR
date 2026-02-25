import numpy as np
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplHistCanvas(FigureCanvas):
    """
    Interactive histogram canvas. User can drag vertical lines to set lower/upper bounds.
    Emits range_changed signal when limits are modified.
    """
    range_changed = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None, figsize=(4,2)):
        fig = Figure(figsize=figsize, tight_layout=False)
        fig.patch.set_facecolor('#162a2a')
        super().__init__(fig)
        self.fig = fig
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor('#162a2a')
        self.ax.tick_params(axis='both', colors='#d9f0ec', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#2e6f6f')
        self.ax.xaxis.label.set_color('#d9f0ec')
        self.ax.yaxis.label.set_color('#d9f0ec')
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.15)
        self.ax.set_yscale('log')
        
        self.setParent(parent)
        
        # Internal state for interactive dragging
        self.low_val = 0.0
        self.high_val = 4095.0
        self.dragging = None # None, 'low', or 'high'
        
        # Interactive Lines
        self.low_line = None
        self.high_line = None
        
        # Connect events
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)

    def set_limits(self, low, high):
        """Set the visual limits programmatically (e.g. from controller or resets)"""
        self.low_val = low
        self.high_val = high
        self._update_lines()

    def show_hist(self, counts, bin_edges):
        self.ax.clear()
        self.ax.set_facecolor('#162a2a')
        self.ax.tick_params(axis='both', colors='#d9f0ec', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#2e6f6f')
        self.ax.xaxis.label.set_color('#d9f0ec')
        self.ax.yaxis.label.set_color('#d9f0ec')
        self.ax.set_yscale('log')

        if counts is None or len(counts) == 0:
            self.ax.text(0.5, 0.5, "No histogram", transform=self.ax.transAxes, ha='center', fontsize=9, color='#d9f0ec')
            self.draw()
            return

        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Use plot for fast unfilled line. 
        self.ax.plot(centers, np.where(counts > 0, counts, 1.0), linewidth=0.8, color='#a6f0e6')
        
        # Add/Update the vertical lines
        self.low_line = self.ax.axvline(self.low_val, color='#ff5555', linestyle='--', linewidth=1.5, alpha=0.8)
        self.high_line = self.ax.axvline(self.high_val, color='#55ff55', linestyle='--', linewidth=1.5, alpha=0.8)
        
        self.ax.set_xlabel("Intensity", fontsize=8, color='#d9f0ec')
        self.draw_idle()

    def _update_lines(self):
        if self.low_line:
            self.low_line.set_xdata([self.low_val])
        if self.high_line:
            self.high_line.set_xdata([self.high_val])
        self.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button != 1: return # Left click only
        
        # Determine which line is closer
        dist_low = abs(event.xdata - self.low_val)
        dist_high = abs(event.xdata - self.high_val)
        
        # Threshold for picking (in data coords?) - better to use pixels if needed but data coord works for linear/log scales if handled
        # For simplicity, just pick the closer one
        if dist_low < dist_high:
            self.dragging = 'low'
        else:
            self.dragging = 'high'
        
        self._update_val_from_event(event)

    def _on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax: return
        self._update_val_from_event(event)

    def _on_release(self, event):
        if self.dragging:
            self.dragging = None
            self.range_changed.emit(self.low_val, self.high_val)

    def _update_val_from_event(self, event):
        if event.xdata is None: return
        
        if self.dragging == 'low':
            self.low_val = min(event.xdata, self.high_val - 1)
        elif self.dragging == 'high':
            self.high_val = max(event.xdata, self.low_val + 1)
            
        self._update_lines()
        # Emit immediately for responsive feel
        self.range_changed.emit(self.low_val, self.high_val)
