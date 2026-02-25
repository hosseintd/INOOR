from PyQt5 import QtCore, QtGui, QtWidgets

class RangeSlider(QtWidgets.QWidget):
    lowerValueChanged = QtCore.pyqtSignal(int)
    upperValueChanged = QtCore.pyqtSignal(int)

    def __init__(self, minimum=0, maximum=255, lower=20, upper=235,
                 orientation=QtCore.Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._min = int(minimum)
        self._max = int(maximum)
        self._lower = int(lower)
        self._upper = int(upper)
        self._orientation = orientation
        self._active = None  # 'lower' or 'upper'
        self._handle_radius = 8
        self._bar_thickness = 6
        self.setMinimumHeight(40)

    def sizeHint(self):
        return QtCore.QSize(300, 40)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        left = 10
        right = w - 10
        length = right - left

        # background bar
        bar_y = h // 2
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(80, 120, 160)))  # subtle darker tone for visibility
        rect = QtCore.QRectF(left, bar_y - self._bar_thickness/2, length, self._bar_thickness)
        p.drawRoundedRect(rect, 3, 3)

        # selected range
        if self._max == self._min:
            lower_x = left
            upper_x = right
        else:
            lower_x = left + (self._lower - self._min) / (self._max - self._min) * length
            upper_x = left + (self._upper - self._min) / (self._max - self._min) * length
        p.setBrush(QtGui.QBrush(QtGui.QColor(100, 150, 250, 200)))
        sel_rect = QtCore.QRectF(lower_x, bar_y - self._bar_thickness/2, max(1.0, upper_x - lower_x), self._bar_thickness)
        p.drawRoundedRect(sel_rect, 3, 3)

        # draw handles as triangles
        def draw_triangle(cx, cy):
            r = self._handle_radius
            points = [
                QtCore.QPointF(cx, cy - r),
                QtCore.QPointF(cx - r, cy + r),
                QtCore.QPointF(cx + r, cy + r),
            ]
            p.setBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
            p.drawPolygon(QtGui.QPolygonF(points))

        draw_triangle(lower_x, bar_y - 2)
        draw_triangle(upper_x, bar_y - 2)

        # draw numeric labels
        p.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220)))
        p.setFont(QtGui.QFont("Arial", 9))
        p.drawText(5, 12, f"{self._min}")
        p.drawText(w - 60, 12, f"{self._max}")
        p.drawText(int(lower_x) - 14, bar_y + 22, f"{self._lower}")
        p.drawText(int(upper_x) - 14, bar_y + 22, f"{self._upper}")

    def _pos_to_value(self, x):
        left = 10
        right = self.width() - 10
        x = max(left, min(right, x))
        fraction = (x - left) / (right - left) if (right - left) != 0 else 0.0
        val = round(self._min + fraction * (self._max - self._min))
        return int(val)

    def mousePressEvent(self, ev):
        x = ev.pos().x()
        left = 10
        right = self.width() - 10
        length = right - left
        lower_x = left + (self._lower - self._min) / (self._max - self._min) * length if (self._max - self._min) != 0 else left
        upper_x = left + (self._upper - self._min) / (self._max - self._min) * length if (self._max - self._min) != 0 else right
        # choose which handle is closer (tie -> upper)
        if abs(x - lower_x) <= abs(x - upper_x):
            self._active = 'lower'
        else:
            self._active = 'upper'
        self.mouseMoveEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._active is None:
            return
        val = self._pos_to_value(ev.pos().x())
        if self._active == 'lower':
            val = max(self._min, min(val, self._upper - 1))
            if val != self._lower:
                self._lower = val
                self.lowerValueChanged.emit(self._lower)
                self.update()
        elif self._active == 'upper':
            val = min(self._max, max(val, self._lower + 1))
            if val != self._upper:
                self._upper = val
                self.upperValueChanged.emit(self._upper)
                self.update()

    def mouseReleaseEvent(self, ev):
        self._active = None

    # properties
    def lower(self):
        return int(self._lower)
    def upper(self):
        return int(self._upper)
    def setLower(self, v):
        v = int(v)
        self._lower = max(self._min, min(v, self._upper - 1)); self.update()
    def setUpper(self, v):
        v = int(v)
        self._upper = min(self._max, max(v, self._lower + 1)); self.update()

    def setRange(self, minimum, maximum):
        """Set the absolute slider range (min,max) and clamp handles to it."""
        self._min = int(minimum)
        self._max = int(maximum)
        if self._lower < self._min:
            self._lower = self._min
        if self._upper > self._max:
            self._upper = self._max
        self.update()
